

if __name__ == '__main__':
    from core.methods.causal_trace.casual_trace import Llama2InferencerWithCorruption, collect_embedding_std, format_question, calculate_hidden_flow
    import torch
    import numpy as np
    import os
    from utils import read_json, load_llama_model_and_tokenizer, model_name_mapping, write_to_json, get_model_name_mapping

    import argparse
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], help='Choose model size (7 or 13)')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--use_docker',default=False,action='store_true')
    args = parser.parse_args()

    model_size = args.model_size
    model_size = str(model_size) + 'b'
    debug = False
    use_docker = args.use_docker
    question_key = 'when_false_premise_question'
    subject_key = 'subject_title'
    score_key = 'entropy_over_concepts'
    ground_truth_key = 'who_question_ground_truth'

    dataset_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/uncertainty/books/llama2-{model_size}-chat_on_books_base_entropy.json"
    result_dir = '/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/books'

    if use_docker:
        dataset_file = f"/mnt/userdata/projects/HalluInducing/results/uncertainty/books/llama2-{model_size}-chat_on_books_base_entropy.json"
        result_dir = f'/mnt/userdata/projects/HalluInducing/results/causal_trace/books/{model_size}'

    samples = read_json(dataset_file)
    samples = samples if not debug else samples[:2]
    print(f"Load dataset from {dataset_file}")

    model_name_mapping = get_model_name_mapping(use_docker)
    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    print("Model Name:", model_name_or_path)

    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)
    inferencer = Llama2InferencerWithCorruption(model, tokenizer)

    print("Calculating subject embedding noise level...")
    noise_level = 3 * collect_embedding_std(inferencer,[sample[subject_key] for sample in samples])
    print(f"Using noise level {noise_level}")

    for i,sample in enumerate(samples):
        if i in list(range(200)):
            continue
        sample_name = sample[subject_key].replace('/','').replace(" ",'_')
        for kind in ["mlp", "self_attn",None]:
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/{i}_{sample_name}{kind_suffix}.npz"
            print(f"Preparing filename {filename}...")
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    inferencer,
                    format_question(sample[question_key], instruction=""),
                    sample[subject_key],
                    sample[score_key],
                    sample[ground_truth_key],
                    kind=kind,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                np.savez(filename, **numpy_result)
                print(f"Saving result to {filename}!")
            else:
                numpy_result = np.load(filename, allow_pickle=True)

    print("Finished Running!")

