import matplotlib.pyplot as plt

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def plot_activation_acc(all_acc,title,fig_size=(8,8),save_path=None):
    # Increase the size of the figure
    plt.figure(figsize=fig_size)  # Adjust the figure size as needed

    plt.plot(all_acc)
    plt.xlabel("Layers")
    plt.ylabel("Accs")
    curr_num_layers = [f"layer {i}" for i in range(all_acc.shape[0])]
    plt.xticks(np.arange(len(curr_num_layers)), curr_num_layers,rotation='vertical')

    # Show the plot
    plt.title(title)
    plt.show()
    if save_path:
        plt.savefig(save_path)

def plot_attention_acc(attention_acc,title,fig_size=(8,8),save_path=None):

    # Increase the size of the figure
    plt.figure(figsize=fig_size)  # Adjust the figure size as needed

    # Create a heatmap
    plt.imshow(attention_acc, cmap='Blues', aspect='auto')
    # plt.imshow(word_attention, cmap='Blues', aspect='auto')

    # Set labels for x and y axes
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    # curr_num_layers = [f"layer {i}" for i in range(attention_acc.shape[0],0,-1)]
    curr_num_layers = [f"layer {i+1}" for i in range(attention_acc.shape[0])]
    plt.xticks(np.arange(len(curr_num_layers)), curr_num_layers)

    # Add a colorbar to show the scale of values
    plt.colorbar()

    # Show the plot
    plt.title(title)
    plt.show()
    if save_path:
        plt.savefig(save_path)


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

    # select some arguments
    parser = argparse.ArgumentParser(description='A simple program with argument parsing.')

    # Add arguments
    parser.add_argument('--model_size', type=int, choices=[7, 13], default=7,help='Choose model size (7 or 13)')
    parser.add_argument('--test_size',type=float,default=0.5,help='choose the test set size')
    parser.add_argument('--random_seed',type=int,default=0,help='choose the random seed')
    parser.add_argument('--max_iter',type=int,default=1000,help='max iteration of the logistic regression classifier')
    parser.add_argument('--num_head_to_intervene',type=int,default=48,help='K, number of top heads to intervene on')

    # Parse the arguments
    args = parser.parse_args()

    # select some arguments
    model_size = str(args.model_size)+'b'
    model_name = "llama2-{}-chat".format(model_size)
    model_name_or_path = model_name_mapping[model_name]
    config = AutoConfig.from_pretrained(model_name_or_path)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    test_size = args.test_size
    random_seed = args.random_seed
    num_heads_to_intervene = args.num_head_to_intervene

    result_path = '/home/zhuoran/hongbang/projects/HalluInducing/results/activation/nobel_prize'

    activation_file = f'{result_path}/{model_name}_feature_head_wise.npy'
    print(f"Reading head wise activations from {activation_file}")
    activations = np.load(activation_file)

    label_file = f'{result_path}/{model_name}_labels.npy'
    print(f"Reading labels from {label_file}")
    labels = np.load(label_file)

    # split the train and test set
    X_train_indices, X_test_indices, y_train, y_test = train_test_split(
        np.arange(0,activations.shape[0]),labels,test_size=test_size,random_state=random_seed
    )
    X_train_all = activations[X_train_indices,:,:]
    X_test_all = activations[X_test_indices,:,:]

    # train the probes using logistic regression
    all_accs = []
    probes = []
    for layer in tqdm(range(num_layers)):
        X_train = X_train_all[:, layer, :]
        X_test = X_test_all[:, layer, :]

        clf = LogisticRegression(random_state=random_seed, max_iter=1000).fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        all_accs.append(accuracy_score(y_test, y_test_pred))
        probes.append(clf)

    all_accs = np.array(all_accs)
    all_accs = all_accs.reshape(num_layers)

    plot_activation_acc(all_accs,title=f"{model_name}",fig_size=(10,5))

    # top_accs = np.argsort(all_accs.reshape(num_heads*num_layers))[::-1][:num_heads_to_intervene]
    # top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    #
    # # print("Top Heads: ", sorted(top_heads))
    # plot_attention_acc(attention_acc=all_head_accs_np,title='Hallu Prediction {}'.format(model_name))


