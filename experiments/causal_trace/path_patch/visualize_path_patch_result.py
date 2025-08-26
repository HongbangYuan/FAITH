import matplotlib.font_manager as fm
fm.fontManager.addfont('/home/zhuoran/hongbang/TimesNewRoman/TIMES.TTF')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np
import os

def plot_result(scores,save_path=None):
    # difference:(n_layers,num_heads)
    fig = plt.figure(figsize=(9,8))# Adjust the figure size as needed
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(axis='both', which='major', labelsize=30)
    im = plt.imshow(scores, cmap='Blues', aspect='auto',vmin=0)
    # plt.colorbar()
    plt.xlabel("Attention Heads",fontsize=35)
    plt.ylabel("Layers",fontsize=35)
    # plt.title("Influence of Attention Heads",fontsize=37)

    # ax = fig.add_subplot(1, 2, 2)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # plt.imshow(scores, cmap='Blues', aspect='auto')

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=28)
    # plt.xlabel("Attention Heads",fontsize=18)
    # plt.ylabel("Layers",fontsize=18)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def get_files_except_dicts(directory):
    files = [f for f in os.listdir(directory) if is_file(os.path.join(directory, f))]
    return files

def is_file(filepath):
    return os.path.isfile(filepath)


def plot_attn_weight(attn_weight,word_list,title=None,fig_size=(9,8),save_path=None):
    assert attn_weight.shape[0] == attn_weight.shape[1] == len(word_list)
    # Increase the size of the figure
    plt.figure(figsize=fig_size)  # Adjust the figure size as needed

    # Create a heatmap
    im = plt.imshow(attn_weight, cmap='Blues', aspect='auto')
    # plt.imshow(word_attention, cmap='Blues', aspect='auto')

    # Set labels for x and y axes
    # plt.xlabel('Words')
    # plt.ylabel('Words')

    # Add a colorbar to show the scale of values
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=28)
    y_labels = word_list
    x_labels = word_list
    plt.xticks(np.arange(len(x_labels)),x_labels,rotation=90,fontsize=30)
    plt.yticks(np.arange(len(y_labels)), y_labels,fontsize=30)

    # Show the plot
    plt.title(title,fontsize=37)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':

    # visualize attention head
    # result_dir_attn_head = '/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/visualize_attn_head'
    # attn_head_files = get_files_except_dicts(result_dir_attn_head)
    # heads_pos = [(1,15),(2,2),(1,22),(5,15),(8,18)]
    # for attn_head_file in attn_head_files:
    #     if not attn_head_file.startswith('0'):
    #         continue
    #     attn_head_file = os.path.join(result_dir_attn_head,attn_head_file)
    #     result = dict(np.load(attn_head_file,allow_pickle=True))
    #     token_attn_weight = result["token_attn_weight"]
    #     token_list = result["token_list"]
    #     for i,head_pos in enumerate(heads_pos):
    #         if i in [2]: # only plot (2,2) and (5,15)
    #             head_name = "-".join(map(str, head_pos))
    #             plot_attn_weight(token_attn_weight[i],word_list=[tok.lower() for tok in token_list],
    #                              save_path='/home/zhuoran/hongbang/attn_head_1-22.pdf'
    #                              )
    #




    # The first logit distribution figure
    result_dir= '/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/path_patch/Movies/head_contributions'
    files = get_files_except_dicts(result_dir)
    score_threshold = 0.1
    count = 0
    all_scores = []
    for file in files:
        file_path = os.path.join(result_dir, file)
        np_result = dict(np.load(file_path, allow_pickle=True))
        score = np_result["differences"]
        if score.max() < score_threshold:
            count += 1
            continue
        all_scores.append(score)

    all_scores_npy = np.stack(all_scores).mean(axis=0)
    plot_result(all_scores_npy)
    # plot_result(all_scores_npy,save_path='/home/zhuoran/hongbang/attention_head_influence_7b_movie.pdf')
    print("Finished Running!")
