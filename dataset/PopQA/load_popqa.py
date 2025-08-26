from utils import read_json
import pandas as pd


def load_popqa():
    file_path = "/home/zhuoran/hongbang/projects/HalluInducing/dataset/PopQA/popQA.tsv"
    column_names = [
        'id', 'subj', 'prop', 'obj', 'subj_id', 'prop_id', 'obj_id', 's_aliases',
        'o_aliases', 's_uri', 'o_uri', 's_wiki_title', 'o_wiki_title', 's_pop',
        'o_pop', 'question', 'possible_answers'
    ]
    # Read the .tsv file into a pandas DataFrame with specified column names
    df = pd.read_csv(file_path, sep='\t', comment='#', names=column_names, skiprows=1)

    # Convert the DataFrame to a list of dictionaries
    data_list = df.to_dict(orient='records')

    return data_list


if __name__ == '__main__':
    samples = load_popqa()
    samples = [sample for sample in samples if sample["s_pop"]>=1e5]
