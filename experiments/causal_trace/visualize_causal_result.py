import numpy as np


numpy_result = np.load("/home/zhuoran/hongbang/projects/HalluInducing/results/causal_trace/books/200_Fantastic_Mr_Fox_mlp.npz",allow_pickle=True)
plot_result = dict(numpy_result)
differences = plot_result["scores"].mean(axis=-1)
low_score = plot_result["low_score"].mean(axis=-1)
