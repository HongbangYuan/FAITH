import os.path

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

def plot_attn_weight(attn_weight,word_list,title,fig_size=(10,9),save_path=None):
    assert attn_weight.shape[0] == attn_weight.shape[1] == len(word_list)
    # Increase the size of the figure
    plt.figure(figsize=fig_size)  # Adjust the figure size as needed

    # Create a heatmap
    plt.imshow(attn_weight, cmap='Blues', aspect='auto')
    # plt.imshow(word_attention, cmap='Blues', aspect='auto')

    # Set labels for x and y axes
    # plt.xlabel('Words')
    # plt.ylabel('Words')

    # Add a colorbar to show the scale of values
    cbar = plt.colorbar()

    # 设置 colorbar 的标签和字体大小
    cbar.ax.tick_params(labelsize=25)
    y_labels = word_list
    x_labels = word_list
    plt.xticks(np.arange(len(x_labels)),x_labels,rotation=90,fontsize=25)
    plt.yticks(np.arange(len(y_labels)), y_labels,fontsize=25)


    # Show the plot
    # plt.title(title)
    plt.title("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    from tqdm import trange

    save_file = "/home/zhuoran/hongbang/projects/HalluInducing/results/rebuttal/weights_llama_7b_on_movies.npy"
    weights_npy = np.load(save_file)
    word_list = ["Why",'was','the','film','<film>','released in','<false year>',"?"]

    # attn_head = (1,15)
    attn_head = (1,22)
    # attn_head = (5,15)
    # attn_head = (26,31)

    attn_weights = weights_npy[attn_head[0],attn_head[1]]
    plot_attn_weight(attn_weights, word_list=word_list,title=f"Attn Head {attn_head[0]}-{attn_head[1]}",
                     # save_path="/home/zhuoran/hongbang/5-15.pdf"
                     save_path="/home/zhuoran/hongbang/1-22.pdf"
    )

    # save_path = '/home/zhuoran/hongbang/projects/HalluInducing/results/rebuttal/attn_weight_per_head_llama2_7b_on_movies'
    # save_path = '/home/zhuoran/hongbang/projects/HalluInducing/results/rebuttal/attn_weight_per_head_llama2_7b_on_movies_pdf'
    # for layer in trange(32):
    #     for head in range(32):
    #         attn_head = (layer,head)
    #         attn_weights = weights_npy[attn_head[0], attn_head[1]]
    #         save_fig = f'Layer {layer} - Head {head}'
    #         plot_attn_weight(attn_weights, word_list=word_list,
    #                          title=f"Attn Head {attn_head[0]}-{attn_head[1]}",
    #                          save_path=os.path.join(save_path,save_fig))
    # print("Finished Running!")
    #
    # for idx,row in enumerate(attn_weights):
    #     print(word_list[idx],end=' |')
    #     for num in row:
    #         print(f"{num:.2f}",end='|')
    #     print("\n")