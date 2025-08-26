import argparse
import os
import pathlib
import pickle
import torch
from tqdm import tqdm
import os.path
import copy

from utils import load_llama_model_and_tokenizer, get_model_name_mapping, CustomDataset, write_to_json, read_json
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from core.inference.llama_inferencer import remove_prompt, format_question


class Llama2Generator:
    def __init__(self, model_size, result_path, experiment_tag, format_question_instruction=format_question,debug=False,debug_num=2,
                 use_docker=False,generation_kwargs=None):
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
        self.format_question_instruction = format_question_instruction
        self.model_name = model_name
        self.debug=debug
        self.debug_num = debug_num

        self.generation_kwargs = generation_kwargs if generation_kwargs else {}

    def reset_result_path(self, result_path):
        self.result_path = result_path

    def reset_experiment_tag(self, experiment_tag):
        self.experiment_tag = f"{self.model_name}_on_{experiment_tag}"

    def run_generation_per_key(self, samples, input_key, batch_size, num_generations=5, instruction='',skip_most_likely=True):
        if self.debug:
            samples = samples[:self.debug_num]
        samples = copy.deepcopy(samples)
        curr_dataset = CustomDataset(samples, input_key)
        curr_data_loader = DataLoader(curr_dataset, batch_size=batch_size)
        result_file = os.path.join(self.result_path,self.experiment_tag+f".json")
        print("Results will be saved in {}!".format(result_file))

        results = []
        for batch in tqdm(curr_data_loader):
            result = dict()
            prompts = list(map(lambda x: self.format_question_instruction(x, instruction), batch[input_key]))
            encoded_batch = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)

            if not skip_most_likely:
                with torch.no_grad():
                    most_likely_generation_ids = self.model.generate(
                        input_ids=encoded_batch["input_ids"],
                        attention_mask=encoded_batch["attention_mask"],
                        max_new_tokens=256,  # Define the maximum length for decoding
                        num_return_sequences=1,
                        num_beams=5,
                        do_sample=False
                    )
                decoded_output = self.tokenizer.batch_decode(most_likely_generation_ids, skip_special_tokens=True)
                most_likely_generation = [
                    remove_prompt(decoded_output[idx]) for idx in range(len(prompts))
                ]
                result["most_likely_generation"] = most_likely_generation
            generations = []
            for i in tqdm(range(num_generations),leave=False):
                generation_ids = self.model.generate(
                    input_ids=encoded_batch["input_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                    **self.generation_kwargs,
                )
                decoded_output = self.tokenizer.batch_decode(generation_ids, skip_special_tokens=True)
                generation = [
                    remove_prompt(decoded_output[idx]) for idx in range(len(prompts))
                ]
                generations.append(generation)
            result["generations"] = generations
            results.append(result)

        new_samples = []
        for result,sample in zip(results,samples):
            new_samples.append({**result,**sample})

        print("Hello World!")

        # # save the running result
        # for model_answer, sample in zip(model_answers, samples):
        #     sample[output_key] = model_answer
        #
        print("Writing result to {}!".format(result_file))
        write_to_json(new_samples, result_file, default=str)


if __name__ == '__main__':
    from dataset.ToyDataset.Books.load_books import load_fp_books

    model_size = 7
    samples = load_fp_books(model_size)

    generator = Llama2Generator(
        model_size=model_size,
        result_path="/home/zhuoran/hongbang/projects/HalluInducing/results/uncertainty/books",
        experiment_tag="books_multiple_generation",
    )

    generator.run_generation_per_key(
        samples,
        input_key='when_false_premise_question',
        batch_size=1,
    )



