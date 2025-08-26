import pandas as pd
import ast


def split_to_list(text):
    if pd.notnull(text):  # Check if 'answer' column is not NaN
        try:
            l = ast.literal_eval(text)
        except (SyntaxError,ValueError):
            l = [text]
        return l
    else:
        return []


def load_falseqa(split):
    assert split in ['train', 'dev', 'test']
    if split == 'dev':
        split = 'valid'
    file_path = f"/home/zhuoran/hongbang/projects/HalluInducing/dataset/FalseQA/{split}.csv"
    df = pd.read_csv(file_path)
    df["answers"] = df["answer"].apply(split_to_list)

    samples = df.to_dict(orient='records')
    return samples


if __name__ == '__main__':
    from utils import select

    train_samples = load_falseqa(split='train')
    dev_samples = load_falseqa(split='dev')
    test_samples = load_falseqa(split='test')

    # false_premise_test_samples = select(dev_samples,labels=['false presupposition'])
