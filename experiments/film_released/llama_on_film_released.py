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
    import argparse

    load_dotenv()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from dataset.ToyDataset.Movies.load_movies import load_movies,load_film_release
    from utils import select, CustomDataset,write_to_json,read_json,model_name_mapping,load_llama_model_and_tokenizer

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size',type=int,default=8,help='batch_size')
    parser.add_argument('--input_key',default='when_question')
    parser.add_argument('--output_key',default='when_question_model_answer')

    # Parse the arguments
    args = parser.parse_args()

    # select some arguments
    batch_size = args.batch_size
    model_size = str(args.model_size)+'b'
    key = args.input_key
    output_key = args.output_key
    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/{}_on_film_release_dataset_when_question.json'.format(model_name)
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/film_released/{}_on_film_release_dataset_untruthful_instruction.json'.format(model_name)
    print("Model Name:",model_name_or_path)
    print("Result File:",result_file)
    print("input_key:",key)
    print("output_key",output_key)

    samples = load_film_release()

    film_release_dataset = CustomDataset(samples, key)

    # Initialize the DataLoader with your dataset
    data_loader = DataLoader(film_release_dataset, batch_size=batch_size)

    # load models
    model,tokenizer = load_llama_model_and_tokenizer(model_name_or_path)

    model_answers = []
    for batch in tqdm(data_loader):
        # prompts = batch["question"]
        prompts = list(map(lambda x: format_question(x), batch[key]))
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
        sample[output_key] = model_answer

    write_to_json(samples,result_file,default=str)
    print("Writing result to {}".format(result_file))
    print("Finished Running!")

