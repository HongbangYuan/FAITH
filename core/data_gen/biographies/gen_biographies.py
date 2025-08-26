from tqdm import tqdm
import torch

def remove_prompt(generation,text):
    return generation.strip(text).strip()


def inference_with_decode(prompts, model, tokenizer, generation_kwargs):
    encoded_batch = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    try:
        # raise RuntimeError('probability tensor contains either `inf`, `nan` or element < 0')
        with torch.no_grad():
            output = model.generate(
                input_ids=encoded_batch["input_ids"],
                attention_mask=encoded_batch["attention_mask"],
                **generation_kwargs,
            )
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    except RuntimeError as e:
        if 'probability tensor contains either `inf`, `nan` or element < 0' in str(e):
            # 处理这个特定的错误
            print("Error: probability tensor contains either `inf`, `nan` or element < 0")
            # 拆分batch，一条一条的进行处理
            decoded_output = []
            for prompt in prompts:
                single_encoded = tokenizer([prompt],return_tensors="pt").to(model.device)
                with torch.no_grad():
                    single_output = model.generate(
                        input_ids=single_encoded["input_ids"],
                        attention_mask=single_encoded["attention_mask"],
                        **generation_kwargs,
                    )
                decoded_output.append(tokenizer.batch_decode(single_output, skip_special_tokens=True)[0])
        else:
            raise e

    answers = [remove_prompt(answer,prompts[i]) for i,answer in enumerate(decoded_output)]
    return answers


if __name__ == '__main__':
    import argparse
    from dataset.Biographies.load_biographies import load_only_names
    from functools import partial
    import os
    from utils.general_utils import CustomDataset
    from utils.llm_utils import load_model_and_tokenizer
    from utils.io_utils import write_to_json
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7, help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=3, type=int, help='batch_size')
    parser.add_argument('--use_docker', action='store_true')
    parser.add_argument('--num_biographies',type=int,default=5)
    args = parser.parse_args()

    # parse args and set some hyperparameters
    model_size = args.model_size
    use_docker = args.use_docker
    batch_size = args.batch_size
    num_biographies = args.num_biographies
    model_name = "llama2-{}b-chat".format(model_size)
    input_key = 'prompt'
    output_key = f'{model_name}_generated_biography'

    # create result path
    base_dir = '/mnt/userdata/projects/HalluInducing/results' if args.use_docker else '/home/zhuoran/hongbang/projects/HalluInducing/results'
    result_path = f'{base_dir}/gen_biographies'
    if not os.path.exists(result_path):
        print(f"Creating the result path {result_path}...")
        os.makedirs(result_path)
    else:
        print(f"Results will be saved in {result_path}.")
    save_file = f"{result_path}/{model_name}_generated_biographies.json"
    print(f"Results will be saved to {save_file}.")

    # load data
    samples = load_only_names()
    curr_dataset = CustomDataset(samples, input_key)
    curr_data_loader = DataLoader(curr_dataset, batch_size=batch_size)

    # create generation kwargs: Multinomial Sampling Decoding
    # set temperature=1.0 to increase the diversity of the generated answers
    generation_kwargs = {
        "max_new_tokens": 256,
        "num_beams": 1,
        "do_sample": True,
        "temperature": 1.0,
    }
    print("generation kwargs:", generation_kwargs)

    # load model
    print("Loading models...")
    model, tokenizer = load_model_and_tokenizer(model_name, use_docker)

    i = 0
    for batch in tqdm(curr_data_loader):
        prompts = batch[input_key]
        all_answers = []
        for i_biography in tqdm(range(num_biographies),leave=False):
            answers = inference_with_decode(prompts, model, tokenizer, generation_kwargs)
            all_answers.append(answers)
        for idx, sample in enumerate(samples[i: i + batch_size]):
            answers_per_sample = [answers[idx] for answers in all_answers]
            sample[output_key] = answers_per_sample
        i += batch_size
        write_to_json(samples,save_file)

    print("Finished Running")