import torch

llama_prompt_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{}{} [/INST]"""

def format_question(question):
    return llama_prompt_template.format(question.capitalize() + '?','')

def format_question_with_prompt(question,prompt):
    return llama_prompt_template.format(question.capitalize() + '?', prompt)

def load_film():
    time = [str(t) for t in range(1900, 2100)]
    film_name = "Ragtime"
    samples = [
        {
            # "question":"When was the {} film {} released?".format(t,film_name)
            "question": "Why was the film {} released in {}".format(film_name, t)
        }
        for t in time
    ]
    return samples

def remove_prompt(text, prompt):
    # return text.lstrip(prompt).strip()
    substring = '[/INST]'
    start_idx = text.find(substring)
    end_index = start_idx + len(substring)
    if start_idx == -1:
        raise ValueError("[/INST] not found in {}!".format(text))
    return text[end_index:].strip()


if __name__ == '__main__':
    from dataset.QAQA.load_qaqa import load_qaqa
    from utils import model_name_mapping,CustomDataset,load_llama_model_and_tokenizer,write_to_json
    import numpy as np
    from torch.utils.data import DataLoader
    from core.methods.repe import repe_pipeline_registry
    from core.methods.repe.rep_control_reading_vec import WrappedReadingVecModel
    from tqdm import tqdm

    repe_pipeline_registry()


    # select some arguments
    batch_size = 1
    # model_size = '7b'
    model_size = '13b'
    coeff = 0.25
    key = 'question'

    # automatically constructing some arguments
    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/repe/{}_on_film_released.json'.format(model_name)
    if model_size == '7b':
        layer_ids = np.arange(8, 32, 3)  # for 7B model
    elif model_size == '13b':
        layer_ids = np.arange(10, 40, 3)  # for 13B model
    else:
        raise ValueError("Model size {} not defined!".format(model_size))
    templates = [
        ('Give a truthful answer. ', 'Give an untruthful answer. '),
    ]
    block_name = "decoder_block"

    print("model_name_or_path:",model_name_or_path)
    print("result_file:",result_file)

    # load dataset
    samples = load_film()
    toy_dataset = CustomDataset(samples)
    data_loader = DataLoader(toy_dataset, batch_size=batch_size)

    # load models
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    # create wrapped model
    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    # make sure nothing is wrapped from previous runs
    wrapped_model.unwrap()
    # wrap model at desired layers and blocks
    wrapped_model.wrap_block(layer_ids, block_name=block_name)

    model_answers = []
    for batch in tqdm(data_loader):
        prompts = list(map(lambda x: format_question(x), batch[key]))
        encoded_batch = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
        # remove token_type_ids
        if "token_type_ids" in encoded_batch:
            del encoded_batch["token_type_ids"]

        directions = {}
        for layer_id in layer_ids:
            directions[layer_id] = 0

        for (experimental_prompt, reference_prompt) in templates:
            wrapped_model.reset()
            batch_pos = list(map(lambda x: format_question_with_prompt(x,experimental_prompt), batch[key]))
            batch_neg = list(map(lambda x: format_question_with_prompt(x,reference_prompt), batch[key]))

            encoded_batch_pos = tokenizer(batch_pos, padding=True, return_tensors="pt").to(model.device)
            encoded_batch_neg = tokenizer(batch_neg, padding=True, return_tensors="pt").to(model.device)

            split = 4
            for layer_id in layer_ids:
                _ = wrapped_model(**encoded_batch_pos)
                pos_outputs = wrapped_model.get_activations(layer_ids, block_name=block_name)
                _ = wrapped_model(**encoded_batch_neg)
                neg_outputs = wrapped_model.get_activations(layer_ids, block_name=block_name)
                directions[layer_id] += coeff * (
                        pos_outputs[layer_id][:, -split:] - neg_outputs[layer_id][:, -split:]) / len(templates)

                wrapped_model.reset()
                wrapped_model.set_controller([l for l in layer_ids if l <= layer_id], directions,
                                             masks=encoded_batch["attention_mask"][:,-split:,None],
                                             token_pos="end",
                                             normalize=False)

        with torch.no_grad():
            output = wrapped_model.generate(
                input_ids=encoded_batch["input_ids"],
                attention_mask=encoded_batch["attention_mask"],
                max_new_tokens=250,  # Define the maximum length for decoding
                repetition_penalty=1.5,
                do_sample=False,
                use_cache=False
            )
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
            answers = [
                remove_prompt(decoded_output[idx], prompts[idx]) for idx in range(len(prompts))
            ]
            model_answers.extend(answers)

    # save the running result
    for model_answer, sample in zip(model_answers, samples):
        sample["model_answer"] = model_answer

    write_to_json(samples, result_file, default=str)
    print("Writing result to {}".format(result_file))
    print("Finished Running!")




