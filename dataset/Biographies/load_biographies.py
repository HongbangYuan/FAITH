import pandas as pd


def load_only_names(use_docker=False):
    filepath = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/Biographies/PopularPeople.csv'
    if use_docker:
        filepath = '/mnt/userdata/projects/HalluInducing/dataset/Biographies/PopularPeople.csv'
    df = pd.read_csv(filepath)
    selected_columns = ['name', 'frame', 'popularity','prompt']
    df['prompt'] = df['name'].apply(lambda x: f"Write a short biography of {x}.")
    df_selected = df[selected_columns]
    return df_selected.to_dict(orient='records')

if __name__ == '__main__':
    samples = load_only_names()