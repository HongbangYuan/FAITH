import pandas as pd
import numpy as np

def convert_ndarray_to_list(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_ndarray_to_list(item) for item in obj)
    else:
        return obj

if __name__ == '__main__':

    # 读取Parquet文件
    df = pd.read_parquet('/home/zhuoran/hongbang/projects/HalluInducing/dataset/WildBench/test-00000-of-00001.parquet')

    # 打印数据框的前几行
    print(df.head())

    samples = df.apply(lambda row: convert_ndarray_to_list(row.to_dict()), axis=1).tolist()
    for sample in samples:
        for single_turn in sample["conversation_input"]:
            if single_turn["language"] != 'English':
                print(single_turn)
