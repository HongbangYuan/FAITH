import os.path
import copy

from utils import load_llama_model_and_tokenizer, get_model_name_mapping, CustomDataset, write_to_json, read_json
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

llama_prompt_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{} [/INST]"""


def format_question(question,instruction):
    return llama_prompt_template.format(instruction+question)


def remove_prompt(text):
    substring = '[/INST]'
    start_idx = text.find(substring)
    end_index = start_idx + len(substring)
    if start_idx == -1:
        raise ValueError("[/INST] not found in {}!".format(text))
    return text[end_index:].strip()


class Llama2Inferencer:
    def __init__(self, model_size,result_path,experiment_tag,format_question_instruction=format_question,use_docker=False,generation_kwargs=None):
        model_name_mapping = get_model_name_mapping(use_docker)
        model_size = str(model_size)+'b'
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

        self.generation_kwargs = generation_kwargs if generation_kwargs else {}

    def inference_with_decode(self,prompts):
        encoded_batch = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids=encoded_batch["input_ids"],
                attention_mask=encoded_batch["attention_mask"],
                **self.generation_kwargs,
            )
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        answers = [
            remove_prompt(decoded_output[idx]) for idx in range(len(prompts))
        ]
        return answers

    def detect_knowledge(self,samples,input_key,output_key,batch_size,instruction,judge_sample):
        assert os.path.exists(self.result_path)
        result_file = os.path.join(self.result_path, self.experiment_tag + f"_{output_key}.json")
        print(f"Result will be write to {result_file}...")
        curr_dataset = CustomDataset(samples,input_key)
        curr_data_loader = DataLoader(curr_dataset,batch_size=batch_size)

        i = 0
        model_answers = []
        true_count = 0
        pbar = tqdm(curr_data_loader)
        for batch in pbar:
            prompts = list(map(lambda x: self.format_question_instruction(x, instruction), batch[input_key]))
            answers = self.inference_with_decode(prompts)

            for idx,sample in enumerate(samples[i : i+batch_size]):
                sample[output_key] = answers[idx]
                true_count += judge_sample(sample)
            pbar.set_description(f"True:{true_count}/{i+batch_size}")
            i += batch_size
            model_answers.extend(answers)
            write_to_json(samples[:i], result_file, default=str)

        print("Writing result to {}!".format(result_file))

    def reset_result_path(self,result_path):
        self.result_path = result_path

    def reset_experiment_tag(self,experiment_tag):
        self.experiment_tag = f"{self.model_name}_on_{experiment_tag}"

    def run_experiment_per_key(self, samples, input_key, output_key, batch_size,instruction,skipt_output_key=False):
        assert os.path.exists(self.result_path)
        result_file = os.path.join(self.result_path, self.experiment_tag + f"_{output_key}.json")
        print(f"Result will be write to {result_file}...")

        # samples = copy.deepcopy(samples)
        curr_dataset = CustomDataset(samples,input_key)
        curr_data_loader = DataLoader(curr_dataset,batch_size=batch_size)

        i = 0
        for batch in tqdm(curr_data_loader):
            prompts = list(map(lambda x: self.format_question_instruction(x,instruction), batch[input_key]))
            answers = self.inference_with_decode(prompts)
            for idx,sample in enumerate(samples[i : i+batch_size]):
                sample[output_key] = answers[idx]
            i += batch_size
            write_to_json(samples[:i], result_file, default=str)

        print("Writing result to {}!".format(result_file))

    def run_experiment(self, samples, input_keys, output_keys, batch_size,instructions,skip_output_key=False):
        assert len(input_keys) == len(output_keys)
        assert len(input_keys) == len(instructions)
        total = len(input_keys)

        for input_key in input_keys:
            assert input_key in samples[0].keys()

        for i,(input_key, output_key,instruction) in enumerate(zip(input_keys, output_keys,instructions)):
            print(f"{i+1}/{total}-------Input Key:{input_key}---------------Output Key:{output_key}---------------")
            self.run_experiment_per_key(samples, input_key, output_key, batch_size,instruction,skip_output_key)

        print("Finished Running")
