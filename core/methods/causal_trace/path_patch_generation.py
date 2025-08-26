from utils import get_model_name_mapping, load_llama_model_and_tokenizer
from tqdm import tqdm
import torch
import numpy as np
import os
from core.inference.llama_inferencer import remove_prompt, format_question
from core.methods.causal_trace.casual_trace import find_token_range, layername
from collections import defaultdict
from utils.nethook import TraceDict
from functools import partial
from core.methods.semantic_uncertainty.uncertainty_calculator import format_question_answer


def untuple(x):
    return x[0] if isinstance(x, tuple) else x


def get_predictive_entropy_over_concepts_per_sample(log_likelihods, semantic_set_ids):
    aggregated_likelihoods = []
    row = torch.tensor(log_likelihods)
    semantic_set_ids_row = torch.tensor(semantic_set_ids)
    for semantic_set_id in torch.unique(semantic_set_ids_row):
        aggregated_likelihoods.append(torch.logsumexp(row[semantic_set_ids_row == semantic_set_id], dim=0))
    aggregated_likelihoods = torch.tensor(aggregated_likelihoods)
    entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
    return entropy


def get_semantic_ids_per_sample(generations, ground_truth, judge_func):
    flags = [judge_func(g[0], ground_truth) for g in generations]
    semantic_ids_per_sample = []
    count = 2
    for flag in flags:
        if flag:
            semantic_ids_per_sample.append(1)
        else:
            semantic_ids_per_sample.append(count)
            count = count + 1
    return semantic_ids_per_sample


class AttnHeadSelector:
    def __init__(self, model_size, result_path, experiment_tag, debug=False, debug_num=2, use_docker=False):
        model_name_mapping = get_model_name_mapping(use_docker)
        model_size = str(model_size) + 'b'
        model_name = "llama2-{}-chat".format(model_size)
        model_name_or_path = model_name_mapping[model_name]
        print("Model Name:", model_name_or_path)

        model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
        self.model = model.bfloat16()
        self.tokenizer = tokenizer
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            raise ValueError(f"Result Path {result_path} not existed!")
        self.experiment_tag = f"{model_name}_on_{experiment_tag}"
        self.format_question = format_question
        self.model_name = model_name
        self.debug = debug
        self.debug_num = debug_num

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_hidden_layers = model.config.num_hidden_layers

        self.sampling_generation_kwargs = dict(
            do_sample=True,
            num_return_sequences=1,
            num_beams=1,
            max_length=256,
            temperature=0.5,
            top_p=1.0,
            output_scores=True,
            return_dict_in_generate=True,
        )

    def catch_original_states(
            self,
            question_ref,
            question_counterfactual,
    ):
        model, tokenizer = self.model, self.tokenizer
        layers = [layername(model, L, 'self_attn.o_proj') for L in range(self.num_hidden_layers)]

        prompts = [
            self.format_question(question_ref, ""),
            self.format_question(question_counterfactual, ""),
        ]
        batch_input = self.tokenizer(prompts, return_tensors='pt', padding=True)
        batch_input = {
            key: value.to(model.device) for key, value in batch_input.items()
        }
        _, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0], "".join(question_ref.split()))
        pos = end_of_question - 2

        states = {}

        def store_states(x, layer, states):
            h = untuple(x)
            states[layer] = h[:, pos, :]
            return x

        with torch.no_grad(), TraceDict(
                model,
                layers,
                edit_input=partial(store_states, states=states),
        ) as td:
            model(
                **batch_input,
                output_attentions=True,
            )
        return states,batch_input,pos

    def calculate_path_patch_with_attn_head(
            self,
            batch_input,
            question,
            layer_id,
            head_id,
            pos,
            states,
            num_generations,
            ground_truth,
            judge_func,
    ):
        model, tokenizer = self.model, self.tokenizer
        layers = [layername(model, L, 'self_attn.o_proj') for L in range(self.num_hidden_layers)]
        selected_layer = layername(model, layer_id, 'self_attn.o_proj')
        head_dim = self.head_dim
        dim_start = head_id * head_dim
        dim_end = (head_id + 1) * head_dim

        def path_rep(x, layer):
            h = untuple(x)
            if h.shape[1] == 1:
                return x
            # keeping all the heads frozen to their activations on reference data
            h[0, pos, :] = states[layer][0]
            if layer == selected_layer:
                h[0, pos, dim_start:dim_end] = states[layer][1][dim_start:dim_end]
            return x

        average_neg_log_likelihoods = []
        generations = []
        for i in range(num_generations):
            with torch.no_grad(), TraceDict(
                    model,
                    layers,
                    edit_input=path_rep
            ):
                # only the question reference is needed
                output = model.generate(
                    input_ids=batch_input["input_ids"][0].unsqueeze(0),
                    attention_mask=batch_input["attention_mask"][0].unsqueeze(0),
                    **self.sampling_generation_kwargs,
                )
                decoded_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
                generation_str = remove_prompt(decoded_output[0])
                generations.append(generation_str)

                answer = generation_str
                orig_prompt = format_question_answer(question, answer)
                generation = self.tokenizer(orig_prompt, return_tensors="pt")["input_ids"]
                prompt = generation[:, :generation.shape[-1] - len(self.tokenizer.tokenize(answer))]

                target_ids = generation.clone()
                target_ids[:, :prompt.shape[-1]] = -100
                model_output = self.model(
                    generation.to(self.model.device)[:, :],
                    labels=target_ids.to(self.model.device)[:, :],
                )
                average_neg_log_likelihood = model_output['loss'].item()
            average_neg_log_likelihoods.append(average_neg_log_likelihood)

        average_neg_log_likelihoods_npy = np.array(average_neg_log_likelihoods)

        semantic_ids = get_semantic_ids_per_sample(generations, ground_truth, judge_func)
        predictive_entropy_over_concepts = \
            get_predictive_entropy_over_concepts_per_sample(
                -average_neg_log_likelihoods_npy,
                semantic_ids
            )
        curr_uncertainty_score = predictive_entropy_over_concepts
        return curr_uncertainty_score.item()

    def cal_path_patch_generation_with_attn_head(
            self,
            ground_truth,
            judge_func,
            question_ref,
            question_counterfactual,
            num_generations,
            head_pos,
    ):
        layer_id,head_id = head_pos
        states,batch_input,pos = self.catch_original_states(
            question_ref,
            question_counterfactual,
        )
        curr_uncertainty_score = self.calculate_path_patch_with_attn_head(
            batch_input,
            question_ref,
            layer_id,
            head_id,
            pos,
            states,
            num_generations,
            ground_truth,
            judge_func
        )
        return curr_uncertainty_score

    def cal_path_patch_generation(
            self,
            ground_truth,
            judge_func,
            question_ref,
            question_counterfactual,
            num_generations,
            base_uncertainty_score,
    ):
        states,batch_input,pos = self.catch_original_states(
            question_ref,
            question_counterfactual,
        )
        differences = []
        for layer_id in tqdm(range(int(self.num_hidden_layers // 2))):
            row = []
            for head_id in tqdm(range(self.num_heads), leave=False):
                curr_uncertainty_score = self.calculate_path_patch_with_attn_head(
                    batch_input,
                    question_ref,
                    layer_id,
                    head_id,
                    pos,
                    states,
                    num_generations,
                    ground_truth,
                    judge_func
                )
                difference = curr_uncertainty_score - base_uncertainty_score
                row.append(difference)
            differences.append(row)
        differences = np.array(differences)
        return dict(
            differences=differences,
            base_uncertainty_score=base_uncertainty_score,
        )

    def cal_knock_out_with_singe_head(
            self,
            batch_input,
            question,
            layer_id,
            head_id,
            pos,
            num_generations,
            ground_truth,
            judge_func,
            return_generations=False,
    ):
        model,tokenizer = self.model,self.tokenizer
        if layer_id != -1 and head_id != -1:
            heads_pos = [(layer_id, head_id)]
            layer_to_head = defaultdict(list)
            for elem in heads_pos:
                layer_to_head[layername(model, elem[0], 'self_attn.o_proj')].append(elem[1])
            layers = list(layer_to_head.keys())

            def intervene_head(x, layer):
                h = untuple(x)
                if untuple(x).shape[1] == 1:
                    # Pay close attention when prompt length=1! But I don't think this would happen.
                    return x
                heads = layer_to_head[layer]
                for head in heads:
                    dim_start = head * self.head_dim
                    dim_end = (head + 1) * self.head_dim
                    h[:, pos, dim_start:dim_end] = 0
                return x
            intervene_func= intervene_head
        else:
            intervene_func = None
            layers = []

        average_neg_log_likelihoods = []
        generations = []
        for i in range(num_generations):
            with torch.no_grad(), TraceDict(
                    model,
                    layers,
                    edit_input=intervene_func
            ):
                # only the question reference is needed
                output = model.generate(
                    input_ids=batch_input["input_ids"],
                    attention_mask=batch_input["attention_mask"],
                    **self.sampling_generation_kwargs,
                )
                decoded_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
                generation_str = remove_prompt(decoded_output[0])
                generations.append(generation_str)

            # with torch.no_grad():
                answer = generation_str
                orig_prompt = format_question_answer(question, answer)
                generation = self.tokenizer(orig_prompt, return_tensors="pt")["input_ids"]
                prompt = generation[:, :generation.shape[-1] - len(self.tokenizer.tokenize(answer))]

                target_ids = generation.clone()
                target_ids[:, :prompt.shape[-1]] = -100
                model_output = self.model(
                    generation.to(self.model.device)[:, :],
                    labels=target_ids.to(self.model.device)[:, :],
                )
                average_neg_log_likelihood = model_output['loss'].item()
            average_neg_log_likelihoods.append(average_neg_log_likelihood)


        average_neg_log_likelihoods_npy = np.array(average_neg_log_likelihoods)

        semantic_ids = get_semantic_ids_per_sample(generations, ground_truth, judge_func)
        predictive_entropy_over_concepts = \
            get_predictive_entropy_over_concepts_per_sample(
                -average_neg_log_likelihoods_npy,
                semantic_ids
            )
        curr_uncertainty_score = predictive_entropy_over_concepts.item()
        if return_generations:
            return curr_uncertainty_score,generations,semantic_ids
        return curr_uncertainty_score

    def build_input(
            self,
            question
    ):
        model, tokenizer = self.model, self.tokenizer
        orig_prompts = [format_question(question, "")]
        batch_input = tokenizer(orig_prompts, return_tensors="pt", padding=True)
        batch_input = {
            key: value.to(model.device) for key, value in batch_input.items()
        }
        _, end_of_question = find_token_range(tokenizer, batch_input["input_ids"][0], "".join(question.split()))
        pos = end_of_question - 2
        return batch_input,pos

    def cal_knock_out_attn_generation_with_single_head(
            self,
            ground_truth,
            judge_func,
            question,
            num_generations,
            head_pos,
            return_generations=False,
    ):
        model, tokenizer = self.model, self.tokenizer

        batch_input,pos = self.build_input(question)
        layer_id,head_id = head_pos
        # base_result = self.cal_knock_out_with_singe_head(
        #     batch_input,
        #     question,
        #     -1,
        #     -1,
        #     pos,
        #     num_generations,
        #     ground_truth,
        #     judge_func,
        #     return_generations,
        # )
        result = self.cal_knock_out_with_singe_head(
            batch_input,
            question,
            layer_id,
            head_id,
            pos,
            num_generations,
            ground_truth,
            judge_func,
            return_generations,
        )
        return result


    def cal_knock_out_attn_generation(
            self,
            ground_truth,
            judge_func,
            question,
            num_generations,
            base_uncertainty_score,
    ):
        model, tokenizer = self.model, self.tokenizer
        layers = [layername(model, L, 'self_attn.o_proj') for L in range(self.num_hidden_layers)]

        batch_input,pos = self.build_input(question)

        differences = []
        for layer_id, selected_layer in enumerate(tqdm(layers[:int(self.num_hidden_layers // 2)])):
            row = []
            for head_id, selected_head in enumerate(tqdm(range(self.num_heads), leave=False)):
                curr_uncertainty_score = self.cal_knock_out_with_singe_head(
                    batch_input,
                    question,
                    layer_id,
                    head_id,
                    pos,
                    num_generations,
                    ground_truth,
                    judge_func
                )
                difference = curr_uncertainty_score - base_uncertainty_score
                row.append(difference)
            differences.append(row)
        differences = np.array(differences)
        return dict(
            differences=differences,
            base_uncertainty_score=base_uncertainty_score,
        )
