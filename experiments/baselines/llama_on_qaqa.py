from datasets import Dataset
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from tqdm import tqdm
import json

load_dotenv()
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


llama_prompt_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Question:{} [/INST]"""
# Let's think step by step and answer the following question in 'yes' or 'no':


def format_question(question):
    # return llama_prompt_template.format(question.capitalize() + '?')
    # return llama_prompt_template.format(question.capitalize() + '?')
    return question.capitalize() + '?'

def remove_prompt(text,prompt):
    return text.lstrip(prompt).strip()
    # substring = '[/INST]'
    # start_idx = text.find(substring)
    # end_index = start_idx + len(substring)
    # if start_idx == -1:
    #     raise ValueError("[/INST] not found in {}!".format(text))
    # return text[end_index:].strip()


if __name__ == '__main__':
    from dataset.QAQA.load_qaqa import load_qaqa
    from utils import select, CustomDataset, write_to_json

    # some arguments
    batch_size = 4
    model_name_or_path = '/home/zhuoran/hongbang/huggingface/Llama-2-13b-chat-hf'
    result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/QAQA/llama2-13b-chat_qa_format.json'
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/QAQA/llama2-7b-chat_new_format.json'
    # model_name_or_path = '/home/zhuoran/hongbang/huggingface/Llama-2-13b-chat-hf'
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/QAQA/llama2-13b-chat_yesno_verification_question_including_valid_qs.json'

    print("model_name:",model_name_or_path)
    print("result file:",result_file)

    samples = load_qaqa(split='eval')
    qaqa_dataset = CustomDataset(samples, 'question')
    # qaqa_dataset = CustomDataset(samples, 'yesno_verification_question_including_valid_qs')

    # Initialize the DataLoader with your dataset
    data_loader = DataLoader(qaqa_dataset, batch_size=batch_size)

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
        # prompts = list(map(lambda x: format_question(x), batch["yesno_verification_question_including_valid_qs"]))
        prompts = list(map(lambda x: format_question(x), batch["question"]))
        encoded_batch = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=encoded_batch["input_ids"],
                attention_mask=encoded_batch["attention_mask"],
                max_new_tokens=250,  # Define the maximum length for decoding
                num_beams=5,
                do_sample=False
            )
        output_truncate_prompt = []
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        answers = [
            # decoded_output[idx].lstrip(prompts[idx].lstrip('<s>')).strip('\n') for idx in range(len(prompts))
            remove_prompt(decoded_output[idx],prompts[idx]) for idx in range(len(prompts))
        ]
        model_answers.extend(answers)

    # save the running result
    for model_answer, sample in zip(model_answers, samples):
        sample["model_answer"] = model_answer

    write_to_json(samples, result_file, default=str)
    print("Writing result to {}".format(result_file))
    print("Finished Running!")
