from einops import rearrange
import random
from core.inference.llama_inferencer import format_question
from experiments.information_flow.cal_saliency_score_nobel_prize import format_question_answer
from utils import CustomDataset
from torch.utils.data import DataLoader

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass, use_random_dir,
                           com_directions):
    interventions = {}
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = []
    for layer, head in top_heads:
        if use_center_of_mass:
            direction = com_directions[layer, head, :]
        elif use_random_dir:
            direction = np.random.normal(size=(128,))
        else:
            direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
        direction = direction / np.linalg.norm(direction)
        # activations = tuning_activations[:,layer,head,:] # batch x 128
        # proj_vals = activations @ direction.T
        # proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.o_proj"].append((head, direction.squeeze()))
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.o_proj"] = sorted(
            interventions[f"model.layers.{layer}.self_attn.o_proj"], key=lambda x: x[0])

    return interventions


def lt_modulated_vector_add(head_output, layer_name,interventions,head_dim,start_edit_location='lt'):
    h = untuple(head_output)
    for head, direction in interventions[layer_name]:
        direction_to_add = torch.tensor(direction).to(h.device)
        dim_start = head * head_dim
        dim_end = (head + 1) * head_dim
        if start_edit_location == 'lt':
            h[:, -1,dim_start:dim_end] += args.alpha * direction_to_add
        else:
            h[:, start_edit_location:, dim_start:dim_end] += args.alpha * direction_to_add
    return h

    # head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
    # for head, direction in interventions[layer_name]:
    #     # direction_to_add = torch.tensor(direction).to(args.device)
    #     direction_to_add = torch.tensor(direction).to(head_output.device)
    #     if start_edit_location == 'lt':
    #         head_output[:, -1, head, :] += args.alpha * direction_to_add
    #     else:
    #         head_output[:, start_edit_location:, head, :] += args.alpha * direction_to_add
    # head_output = rearrange(head_output, 'b s h d -> b s (h d)')
    # return head_output


class ItiInferencer:
    def __init__(self, model_size,result_path,experiment_tag,format_question_instruction=format_question,use_docker=False,generation_kwargs=None,rewrite=True):
        model_name_mapping = get_model_name_mapping(use_docker)
        model_size = str(model_size)+'b'
        self.model_size = model_size
        model_name = "llama2-{}-chat".format(model_size)
        model_name_or_path = model_name_mapping[model_name]
        print("Model Name:", model_name_or_path)

        model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
        self.model = model
        self.tokenizer = tokenizer
        self.result_path = result_path
        self.experiment_tag = f"{model_name}_on_{experiment_tag}"
        self.format_question_instruction = format_question_instruction
        self.model_name = model_name
        self.num_hidden_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.generation_kwargs = generation_kwargs if generation_kwargs else {}
        self.rewrite = rewrite


    def get_head_wise_features(self,samples,question_key,answer_key,label_key,head_feature_cache_file,rewrite=False):
        # get activations
        model,tokenizer = self.model,self.tokenizer
        num_hidden_layers,num_heads = self.num_hidden_layers,self.num_heads
        head_wise_activations = []
        labels = []
        layers = [layername(model, L, 'self_attn.o_proj') for L in range(num_hidden_layers)]
        if rewrite or not os.path.isfile(head_feature_cache_file):
            print(f"Constructing cache file {head_feature_cache_file}...")

            for sample in tqdm(samples):
                labels.append(sample[label_key])
                prompt = format_question_answer(sample[question_key], sample[answer_key])
                encoded_prompt = tokenizer(prompt, return_tensors="pt").input_ids

                with torch.no_grad(), TraceDict(
                        model,
                        layers=layers,
                        retain_input=True,
                        retain_output=False,
                ) as td:
                    output = model(encoded_prompt.to(model.device), output_attentions=True)
                head_wise_activation_per_sample = np.stack(
                    [td[layer].input.clone().detach().cpu().numpy()[0, -1] for layer in layers]).squeeze()
                head_wise_activations.append(head_wise_activation_per_sample)
            head_wise_activations = np.stack(head_wise_activations)
            results = {
                "head_wise_feature":rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads),
                "labels":np.array(labels),
            }
            np.savez(
                head_feature_cache_file,
                **results
            )
            print(f"Saving head wise feature to file {head_feature_cache_file}.")
        else:
            results = dict(np.load(head_feature_cache_file,allow_pickle=True))
        return results

    def choose_top_head(self,results,test_size,random_seed,max_iter,num_head_to_intervene):
        num_hidden_layers,num_heads = self.num_hidden_layers,self.num_heads
        hidden_size = self.model.config.hidden_size
        head_dim = hidden_size // num_heads
        head_wise_activations = results["head_wise_feature"]
        labels = results["labels"]

        # split the train and test set
        X_train_indices, X_test_indices, y_train, y_test = train_test_split(
            np.arange(0, head_wise_activations.shape[0]), labels, test_size=test_size, random_state=random_seed,
            shuffle=True
        )
        X_train_all = head_wise_activations[X_train_indices, :, :, :]
        X_test_all = head_wise_activations[X_test_indices, :, :, :]

        # train the probes using logistic regression
        all_head_accs = []
        probes = []
        for layer in tqdm(range(num_hidden_layers)):
            for head in range(num_heads):
                X_train = X_train_all[:, layer, head, :]
                X_test = X_test_all[:, layer, head, :]

                clf = LogisticRegression(random_state=random_seed, max_iter=max_iter).fit(X_train, y_train)
                y_test_pred = clf.predict(X_test)
                all_head_accs.append(accuracy_score(y_test, y_test_pred))
                probes.append(clf)

        all_head_accs_np = np.array(all_head_accs)
        all_head_accs_np = all_head_accs_np.reshape(num_hidden_layers, num_heads)

        top_accs = np.argsort(all_head_accs_np.reshape(num_heads * num_hidden_layers))[::-1][:num_head_to_intervene]
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]

        # print("Top Heads: ", sorted(top_heads))
        # plot_attention_acc(attention_acc=all_head_accs_np,title='Hallu Prediction {}'.format(model_name))

        # get the directions
        usable_labels = labels[X_train_indices]
        usable_head_wise_activations = head_wise_activations[X_train_indices, :, :, :]
        directions = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0) - np.mean(
            usable_head_wise_activations[usable_labels == 0], axis=0)

        # get the intervention dict
        interventions = get_interventions_dict(top_heads, probes, None, num_heads, use_center_of_mass=True,
                                               use_random_dir=False, com_directions=directions)
        intervene = partial(lt_modulated_vector_add,interventions=interventions,head_dim=head_dim,start_edit_location='lt')
        layers_to_intervene = list(interventions.keys())

        return intervene,layers_to_intervene

    def run_experiment(self,samples,input_keys,answer_keys,answer_eval_keys,output_keys,batch_size):
        model,tokenizer = self.model,self.tokenizer
        total = len(input_keys)
        for i,(input_key, answer_key,answer_eval_key,output_key) in enumerate(zip(input_keys,answer_keys,answer_eval_keys,output_keys)):
            print(f"{i+1}/{total}-------Input Key:{input_key}---------------Output Key:{output_key}---------------")
            result_file = os.path.join(self.result_path, self.experiment_tag + f"_{output_key}.json")
            print(f"Result will be write to {result_file}...")

            head_feature_cache_file = f'{self.result_path}/llama2-chat-{self.model_size}_{input_key}.npz'

            results = self.get_head_wise_features(
                samples,
                question_key=input_key,
                answer_key=answer_key,
                head_feature_cache_file=head_feature_cache_file,
                label_key=answer_eval_key,
                rewrite=args.rewrite_cache
            )
            intervene,layers_to_intervene = inferencer.choose_top_head(
                results,
                test_size,
                random_seed,
                max_iter,
                num_head_to_intervene,
            )
            curr_dataset = CustomDataset(samples, input_key)
            data_loader = DataLoader(curr_dataset, batch_size=batch_size)

            with torch.no_grad():
                model_answers = []
                for batch in tqdm(data_loader):
                    prompts = list(map(lambda x: format_question(x, instruction=""),
                                       batch[input_key]))
                    encoded_batch = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

                    with TraceDict(model, layers_to_intervene, edit_input=intervene) as ret:
                        output = model.generate(
                            input_ids=encoded_batch["input_ids"],
                            attention_mask=encoded_batch["attention_mask"],
                            **self.generation_kwargs,
                        )
                    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
                    answers = [
                        # decoded_output[idx].lstrip(prompts[idx].lstrip('<s>')).strip('\n') for idx in range(len(prompts))
                        remove_prompt(decoded_output[idx]) for idx in range(len(prompts))
                    ]
                    model_answers.extend(answers)
            # save the running result
            for model_answer, sample in zip(model_answers, samples):
                sample[output_key] = model_answer

            print(f"Writing result to {result_file}")
            write_to_json(
                samples,
                result_file
            )

if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from collections import defaultdict
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    import argparse
    import torch
    from utils import read_json, load_llama_model_and_tokenizer, get_model_name_mapping, write_to_json, \
        load_llama_tokenizer, min_indices, select
    from utils.nethook import TraceDict
    from tqdm import tqdm
    from core.methods.information_flow.saliency_score import remove_prompt
    from core.methods.causal_trace.causal_trace_tok_pred import layername
    from collections import defaultdict
    from core.methods.causal_trace.casual_trace import find_token_range, untuple
    from core.evaluation.noble_prize.nobel_prie_when_fp_evaluation import cal_when_fp_acc
    import os
    import argparse
    import numpy as np
    from functools import partial

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--use_docker', action='store_true')
    parser.add_argument('--rewrite_cache',action='store_true',default=False)
    parser.add_argument('--test_size', type=float, default=0.5, help='choose the test set size')
    parser.add_argument('--random_seed', type=int, default=0, help='choose the random seed')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='max iteration of the logistic regression classifier')
    parser.add_argument('--num_head_to_intervene', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=1, help='intervention strength')
    parser.add_argument('--batch_size', type=int, default=3, help='inference batch size')


    args = parser.parse_args()

    max_iter = args.max_iter
    test_size = args.test_size
    random_seed = args.random_seed
    question_key = 'fp_question_1'
    answer_key = 'fp_question_1_model_answer'
    answer_eval_key = 'val'

    model_size = f'{args.model_size}b'
    use_docker = args.use_docker
    debug = args.debug
    num_head_to_intervene = args.num_head_to_intervene
    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    result_dir = f'{base_dir}/results/iti/NobelPrize/{model_size}/update_trainset'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Creating result path {result_dir}...")
    # head_feature_cache_file = f'{result_dir}/llama2-chat-{model_size}_{question_key}.npz'

    dataset_file = f'{base_dir}/results/baselines/nobel_prize/llama2-{model_size}-chat_on_nobel_prize_more_fp_when_fp_question4_model_answer.json'
    samples = read_json(dataset_file)

    # Three parameters have to be specified before running
    # input_keys: the question
    # answer_keys: the model answer to the question without any intervene, used to construct the train set and test set
    # answer_eval_keys: whether the model answer is right or wrong, used to construct the train labels and test labels
    input_keys = ['when_fp_question','when_fp_question2','when_fp_question3','when_fp_question4']
    answer_keys = [input_key+'_model_answer' for input_key in input_keys]
    answer_eval_keys = [input_key+'_model_answer_eval' for input_key in input_keys]

    for input_key,answer_key,answer_eval_key in zip(input_keys,answer_keys,answer_eval_keys):
        acc = cal_when_fp_acc(samples,answer_key,answer_eval_key)
        print(f"{input_key} Acc:{acc:.4f}")

    for input_key in input_keys:
        assert input_key in samples[0]
    output_keys = [k+'_iti_answer' for k in input_keys]

    generation_kwargs = {
        "max_length": 256,
        "num_beams": 5,
        "do_sample": False,
    }
    print("generation kwargs:", generation_kwargs)

    inferencer = ItiInferencer(
        args.model_size,
        result_path=result_dir,
        experiment_tag="nobel_prize_iti",
        use_docker=use_docker,
        generation_kwargs=generation_kwargs,
        rewrite=args.rewrite_cache,
    )

    inferencer.run_experiment(
        samples,
        input_keys,
        answer_keys,
        answer_eval_keys,
        output_keys,
        batch_size=args.batch_size
    )

    print("Finished Running!")