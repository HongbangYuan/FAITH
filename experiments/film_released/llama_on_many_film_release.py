from utils import select
import pandas as pd


llama_prompt_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Question:{} [/INST]"""

def format_question(question):
    return llama_prompt_template.format(question)

def remove_prompt(text):
    substring = '[/INST]'
    start_idx = text.find(substring)
    end_index = start_idx + len(substring)
    if start_idx == -1:
        raise ValueError("[/INST] not found in {}!".format(text))
    return text[end_index:].strip()


if __name__ == '__main__':
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from dotenv import load_dotenv
    from tqdm import tqdm
    import json

    load_dotenv()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from dataset.ToyDataset.Movies.load_movies import load_movies
    from utils import select, CustomDataset,write_to_json,read_json,model_name_mapping

    # select some arguments
    batch_size = 8
    # model_size = '7b'
    model_size = '13b'
    key = 'false_premise_question'
    output_key = "false_premise_model_answer"
    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/{}_on_film_release_date.json'.format(model_name)
    print("Model Name:",model_name_or_path)
    print("Result File:",result_file)

    samples = load_movies()

    freshqa_dataset = CustomDataset(samples, 'question')

    # Initialize the DataLoader with your dataset
    data_loader = DataLoader(freshqa_dataset, batch_size=batch_size)

    # load models
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map='balanced',
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'

    model_answers = []
    for batch in tqdm(data_loader):
        # prompts = batch["question"]
        prompts = list(map(lambda x: format_question(x), batch["question"]))
        encoded_batch = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=encoded_batch["input_ids"],
                attention_mask=encoded_batch["attention_mask"],
                max_length=500,  # Define the maximum length for decoding
                num_beams=5,
                do_sample=False
            )
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        answers = [
            # decoded_output[idx].lstrip(prompts[idx].lstrip('<s>')).strip('\n') for idx in range(len(prompts))
            remove_prompt(decoded_output[idx]) for idx in range(len(prompts))
        ]
        model_answers.extend(answers)

    # save the running result
    for model_answer,sample in zip(model_answers,samples):
        sample["model_answer"] = model_answer

    write_to_json(samples,result_file,default=str)
    print("Writing result to {}".format(result_file))
    print("Finished Running!")
