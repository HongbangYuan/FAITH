import pandas as pd


def load_qaqa(split='eval'):
    assert split in ['eval', 'adaption']
    if split == 'eval':
        file_path = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/QAQA/QAQA_evaluation_set_Dec2022.csv'
    else:
        file_path = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/QAQA/QAQA_adaptation_set_Dec2022.csv'
    df = pd.read_csv(file_path, index_col=0)
    return df.to_dict(orient='records')


if __name__ == '__main__':
    from utils import select
    samples = load_qaqa(split='eval')
    sub_samples = select(samples,all_assumptions_valid='has_invalid')
    whose_samples = [sample for sample in samples if sample["question"].startswith("whose")]
    when_samples = [sample for sample in samples if sample["question"].startswith("when")]
    where_samples = [sample for sample in sub_samples if sample["question"].startswith("where")]
    which_samples = [sample for sample in sub_samples if sample["question"].startswith("which")]
    what_samples = [sample for sample in sub_samples if sample["question"].startswith("what")]
    what_samples = [sample for sample in sub_samples if sample["question"].startswith("what")]
