import pandas as pd
from torch.utils.data import Dataset


# Define a function to combine non-empty values from multiple columns into a list
def combine_answers(row):
    answers = [row[f'answer_{i}'] for i in range(10) if pd.notnull(row[f'answer_{i}'])]
    return answers


# Function to split text by '\n' and convert it into a list
def split_text(row):
    if pd.notnull(row['source']):  # Check if 'source' column is not NaN
        return row['source'].split('\n')
    else:
        return []


def load_fresh_qa(split=None):
    if split:
        assert split in ['dev', 'test']

    file_path = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/FreshQA/FreshQA_v11232023.xlsx'
    df = pd.read_excel(file_path, skiprows=2, index_col=0)
    df['effective_year'] = df['effective_year'].astype(str)
    df['next_review'] = df['next_review'].astype(str)

    df['answers'] = df.apply(combine_answers, axis=1)

    # Drop the original 'answer_' columns
    df.drop(columns=[f'answer_{i}' for i in range(10)], inplace=True)

    # Apply the function to create a new 'source_list' column
    df['sources'] = df.apply(split_text, axis=1)

    # Drop the original 'source' column if needed
    df.drop(columns='source', inplace=True)

    # Filter rows based on 'split' column value 'TEST' and 'DEV'
    test_rows = df[df['split'] == 'TEST'].to_dict(orient='records')
    # test_rows = df[df['split'] == 'TEST']
    dev_rows = df[df['split'] == 'DEV'].to_dict(orient='records')
    # dev_rows = df[df['split'] == 'DEV']

    if not split:
        return {
            'test': test_rows,
            'dev': dev_rows
        }
    else:
        if split == 'test':
            return test_rows
        else:
            return dev_rows


class FreshQADataset(Dataset):
    def __init__(self, split=None):
        self.data = load_fresh_qa(split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    from utils import select

    samples = load_fresh_qa(split='test')
    false_premise_samples = select(samples, false_premise=True, num_hops='multi-hop')
