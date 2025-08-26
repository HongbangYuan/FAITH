from core.methods.repe import repe_pipeline_registry
from core.methods.repe.rep_control_reading_vec import WrappedReadingVecModel
from utils import load_llama_model_and_tokenizer, model_name_mapping, CustomDataset, write_to_json, read_json
import numpy as np
from tqdm import tqdm
import copy
from torch.utils.data import DataLoader
import torch
import os
from functools import partial
from utils import get_model_name_mapping

repe_pipeline_registry()

llama_prompt_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{} [/INST]"""


def format_question(question, instruction, question_first=False):
    if question_first:
        return llama_prompt_template.format(question + instruction)
    else:
        return llama_prompt_template.format(instruction + question)


def remove_prompt(text):
    substring = '[/INST]'
    start_idx = text.find(substring)
    end_index = start_idx + len(substring)
    if start_idx == -1:
        raise ValueError("[/INST] not found in {}!".format(text))
    return text[end_index:].strip()


class Llama2RepeInferencer:
    def __init__(self, model_size, result_path, experiment_tag,coeff=0.25,use_docker=False):
        model_size = str(model_size) + 'b'
        if model_size == '7b':
            layer_ids = np.arange(8, 32, 3)  # for 7B model
        elif model_size == '13b':
            layer_ids = np.arange(10, 40, 3)  # for 13B model
        else:
            raise ValueError("Model size {} not defined!".format(model_size))
        model_name = "llama2-{}-chat".format(model_size)
        model_name_mapping = get_model_name_mapping(use_docker)
        model_name_or_path = model_name_mapping[model_name]
        block_name = "decoder_block"
        print("Model Name:", model_name_or_path)

        # load models
        model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
        # create wrapped model
        wrapped_model = WrappedReadingVecModel(model, tokenizer)
        # make sure nothing is wrapped from previous runs
        wrapped_model.unwrap()
        # wrap model at desired layers and blocks
        wrapped_model.wrap_block(layer_ids, block_name=block_name)

        self.model = model
        self.wrapped_model = wrapped_model
        self.tokenizer = tokenizer
        self.result_path = result_path
        self.experiment_tag = f"{model_name}_on_{experiment_tag}"
        self.format_question_instruction = partial(format_question,question_first=True)
        self.templates = [
            ('Give a truthful answer. ', 'Give an untruthful answer. '),
        ]
        self.layer_ids = layer_ids
        self.block_name = block_name
        self.coeff = coeff
        self.model_name = model_name

    def reset_result_path(self, result_path):
        self.result_path = result_path

    def reset_experiment_tag(self, experiment_tag):
        self.experiment_tag = f"{self.model_name}_on_{experiment_tag}"

    def run_experiment(self, samples, input_keys, output_keys, batch_size, instructions):
        assert len(input_keys) == len(output_keys)
        assert len(input_keys) == len(instructions)
        total = len(input_keys)

        for input_key in input_keys:
            assert input_key in samples[0].keys()

        for i, (input_key, output_key, instruction) in enumerate(zip(input_keys, output_keys, instructions)):
            print(f"{i + 1}/{total}-------Input Key:{input_key}---------------Output Key:{output_key}---------------")
            self.run_experiment_per_key(samples, input_key, output_key, batch_size, instruction)

        print("Finished Running")

    def run_experiment_per_key(self, samples, input_key, output_key, batch_size, instruction):
        # samples = copy.deepcopy(samples)
        curr_dataset = CustomDataset(samples, input_key)
        curr_data_loader = DataLoader(curr_dataset, batch_size=batch_size)

        model_answers = []
        for batch in tqdm(curr_data_loader):
            prompts = list(map(lambda x: self.format_question_instruction(x,instruction), batch[input_key]))
            encoded_batch = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
            # remove token_type_ids
            if "token_type_ids" in encoded_batch:
                del encoded_batch["token_type_ids"]

            directions = {}
            for layer_id in self.layer_ids:
                directions[layer_id] = 0

            for (experimental_prompt, reference_prompt) in self.templates:
                self.wrapped_model.reset()
                batch_pos = list(map(lambda x: self.format_question_instruction(x, experimental_prompt), batch[input_key]))
                batch_neg = list(map(lambda x: self.format_question_instruction(x, reference_prompt), batch[input_key]))

                encoded_batch_pos = self.tokenizer(batch_pos, padding=True, return_tensors="pt").to(self.model.device)
                encoded_batch_neg = self.tokenizer(batch_neg, padding=True, return_tensors="pt").to(self.model.device)

                split = 1
                for layer_id in self.layer_ids:
                    _ = self.wrapped_model(**encoded_batch_pos)
                    pos_outputs = self.wrapped_model.get_activations(self.layer_ids, block_name=self.block_name)
                    _ = self.wrapped_model(**encoded_batch_neg)
                    neg_outputs = self.wrapped_model.get_activations(self.layer_ids, block_name=self.block_name)
                    directions[layer_id] += self.coeff * (
                            pos_outputs[layer_id][:, -split:] - neg_outputs[layer_id][:, -split:]) / len(self.templates)

                    self.wrapped_model.reset()
                    self.wrapped_model.set_controller([l for l in self.layer_ids if l <= layer_id], directions,
                                                 masks=encoded_batch["attention_mask"][:, -split:, None],
                                                 token_pos="end",
                                                 normalize=False)

            with torch.no_grad():
                output = self.wrapped_model.generate(
                    input_ids=encoded_batch["input_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                    max_length=250,  # Define the maximum length for decoding
                    repetition_penalty=1.5,
                    do_sample=False,
                    use_cache=False
                )
                decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                answers = [
                    remove_prompt(decoded_output[idx]) for idx in range(len(prompts))
                ]
                model_answers.extend(answers)

        # save the running result
        for model_answer, sample in zip(model_answers, samples):
            sample[output_key] = model_answer

        result_file = os.path.join(self.result_path, self.experiment_tag + f"_{output_key}.json")
        print("Writing result to {}!".format(result_file))
        write_to_json(samples, result_file, default=str)
