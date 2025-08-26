import matplotlib.pyplot as plt
import torch


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


def plot_attention_acc(attention_acc, title, fig_size=(8, 8), save_path=None):
    # Increase the size of the figure
    plt.figure(figsize=fig_size)  # Adjust the figure size as needed

    # Create a heatmap
    plt.imshow(attention_acc, cmap='Blues', aspect='auto')
    # plt.imshow(word_attention, cmap='Blues', aspect='auto')

    # Set labels for x and y axes
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    # curr_num_layers = [f"layer {i}" for i in range(attention_acc.shape[0],0,-1)]
    curr_num_layers = [f"layer {i}" for i in range(attention_acc.shape[0])]
    plt.yticks(np.arange(len(curr_num_layers)), curr_num_layers)

    # Add a colorbar to show the scale of values
    plt.colorbar()

    # Show the plot
    plt.title(title)
    plt.show()
    if save_path:
        plt.savefig(save_path)


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


def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'):
    head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
    for head, direction in interventions[layer_name]:
        # direction_to_add = torch.tensor(direction).to(args.device)
        direction_to_add = torch.tensor(direction).to(head_output.device)
        if start_edit_location == 'lt':
            head_output[:, -1, head, :] += args.alpha * direction_to_add
        else:
            head_output[:, start_edit_location:, head, :] += args.alpha * direction_to_add
    head_output = rearrange(head_output, 'b s h d -> b s (h d)')
    return head_output


if __name__ == '__main__':
    import numpy as np
    import argparse
    from utils import model_name_mapping
    from transformers import AutoConfig
    from einops import rearrange
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from tqdm import tqdm
    from core.inference.llama_inferencer import format_question, remove_prompt
    from dataset.ToyDataset.Awards.load_awards import load_nobal_prizes
    from dataset.ToyDataset.Movies.load_movies import load_film_release
    from dataset.ToyDataset.Awards.load_oscar import load_oscar_prizes
    from utils.llm_utils import load_llama_model_and_tokenizer
    from utils import write_to_json
    from baukit import TraceDict
    from functools import partial
    from utils import CustomDataset
    from torch.utils.data import DataLoader

    # select some arguments
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7, help='Choose model size (7 or 13)')
    parser.add_argument('--test_size', type=float, default=0.5, help='choose the test set size')
    parser.add_argument('--random_seed', type=int, default=0, help='choose the random seed')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='max iteration of the logistic regression classifier')
    parser.add_argument('--num_head_to_intervene', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=1, help='intervention strength')
    parser.add_argument('--batch_size', type=int, default=8, help='inference batch size')

    # Parse the arguments
    args = parser.parse_args()

    # select some arguments
    model_size = str(args.model_size) + 'b'
    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    config = AutoConfig.from_pretrained(model_name_or_path)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    test_size = args.test_size
    random_seed = args.random_seed
    num_heads_to_intervene = args.num_head_to_intervene

    result_path = '/home/zhuoran/hongbang/projects/HalluInducing/results/iti/nobel_prizes/activations'

    head_wise_activation_file = f'{result_path}/{model_name}_feature_head_wise.npy'
    print(f"Reading head wise activations from {head_wise_activation_file}")
    head_wise_activations = np.load(head_wise_activation_file)
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)

    label_file = f'{result_path}/{model_name}_labels.npy'
    print(f"Reading labels from {label_file}")
    labels = np.load(label_file)

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
    for layer in tqdm(range(num_layers)):
        for head in range(num_heads):
            X_train = X_train_all[:, layer, head, :]
            X_test = X_test_all[:, layer, head, :]

            clf = LogisticRegression(random_state=random_seed, max_iter=1000).fit(X_train, y_train)
            y_test_pred = clf.predict(X_test)
            all_head_accs.append(accuracy_score(y_test, y_test_pred))
            probes.append(clf)

    all_head_accs_np = np.array(all_head_accs)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)

    top_accs = np.argsort(all_head_accs_np.reshape(num_heads * num_layers))[::-1][:num_heads_to_intervene]
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
    intervene = partial(lt_modulated_vector_add, start_edit_location='lt')
    layers_to_intervene = list(interventions.keys())

    # start the inference process
    # Third Try: multiple inference
    # samples = load_nobal_prizes()
    samples = load_film_release()
    # samples = load_oscar_prizes()
    model, tokenizer = load_llama_model_and_tokenizer(model_name_or_path)

    # input_keys = ['who_question','who_question_false_premise','when_question','where_question','where_question_actually']
    input_keys2instructions = {
        'when_question':''
    }
    input_keys = list(input_keys2instructions.keys())
    for input_key in input_keys:
        assert input_key in samples[0]
    output_keys = [k+'_model_answer' for k in input_keys]
    for input_key,output_key in zip(input_keys,output_keys):
        result_file = f"/home/zhuoran/hongbang/projects/HalluInducing/results/iti/film_release/{model_name}-iti_on_film_release_{input_key}.json"
        print(f"---------input_key={input_key}-------------output_key={output_key}-----------")
        print(f"Result File={result_file}")

        curr_dataset = CustomDataset(samples, input_key)
        data_loader = DataLoader(curr_dataset, batch_size=args.batch_size)

        with torch.no_grad():
            model_answers = []
            for batch in tqdm(data_loader):
                prompts = list(map(lambda x: format_question(x, instruction=input_keys2instructions[input_key]), batch[input_key]))
                encoded_batch = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

                with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret:
                    output = model.generate(
                        input_ids=encoded_batch["input_ids"],
                        attention_mask=encoded_batch["attention_mask"],
                        max_new_tokens=200,  # Define the maximum length for decoding
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
        for model_answer, sample in zip(model_answers, samples):
            sample[output_key] = model_answer

        print(f"Writing result to {result_file}")
        write_to_json(
            samples,
            result_file
        )

    print("Finished Running!")
