import pandas as pd
from utils import read_json,read_jsonl
from utils import select


def read_and_process_movies_metadata(file_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Drop rows with missing or invalid release_date
    df = df.dropna(subset=['release_date'])
    df = df[df['release_date'].str.match(r'\d{4}-\d{2}-\d{2}')]

    # Extract year from release_date and create a new column 'release_year'
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    # Drop rows with missing release_year
    df = df.dropna(subset=['release_year'])

    # Convert release_year to string
    df['release_year'] = df['release_year'].astype(int).astype(str)

    # Select only 'original_title' and 'release_year' columns
    df = df[['original_title', 'release_year']]

    # Add a new column 'question'
    df['question'] = df['original_title'].apply(lambda title: "When was the film {} released".format(title))

    # Convert DataFrame to a list of dictionaries
    movies_list = df.to_dict(orient='records')

    return movies_list


def load_movies(num_sample=5000):
    file_path = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Movies/movies_metadata.csv'
    samples = read_and_process_movies_metadata(file_path)[:num_sample]
    return samples


def load_film_release():
    file_path = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Movies/film_release.json'
    samples = read_json(file_path)
    samples = add_when_question(samples)
    samples = add_why_fp_question(samples)
    return samples


def add_when_question(samples):
    for sample in samples:
        sample["when_question"] = "When was the film {} released?".format(sample["original_title"])
    return samples


def add_why_fp_question(samples):
    for sample in samples:
        sample["why_fp_question"] = sample["false_premise_question"]
    return samples

def load_wikidata_movies():
    filepath = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Movies/wikidata_movies.json'
    samples = read_jsonl(filepath)
    add_when_for_wikidata_movies(samples)
    add_why_fp_for_wikidata_movies(samples)
    return samples

def load_wikidata_movies_knowing(model_size=7,use_docker=False):
    assert model_size in [7,13]
    model_size = str(model_size) + 'b'
    file_path = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_when_question_model_answer.json'
    if use_docker:
        file_path = f'/mnt/userdata/projects/HalluInducing/results/baselines/Movies/llama2-{model_size}-chat_on_wiki_movies_when_question_model_answer.json'
    samples = read_json(file_path)
    add_why_fp_with_director_for_wikidata_movies(samples)
    return select(samples,pred=True)


def add_why_fp_with_director_for_wikidata_movies(samples):
    for sample in samples:
        false_year = sample["false_year"]
        director = sample["director"]
        sample["why_fp_question_with_director"] = "Why was the film {} directed by {} released in {}?".format(sample["movie"],director,false_year)


def add_why_fp_for_wikidata_movies(samples):
    for sample in samples:
        false_year = min(sample["time"]) - 1
        sample["false_year"] = false_year
        sample["why_fp_question"] = "Why was the film {} released in {}?".format(sample["movie"],false_year)

def add_when_for_wikidata_movies(samples):
    for sample in samples:
        sample["when_question"] = "When was the film {} released?".format(sample["movie"])

def add_more_fps(samples):
    for sample in samples:
        false_year = min(sample["time"]) - 1
        sample["fp_question_1"] = sample["why_fp_question"]
        sample["fp_question_2"] = f"What was the film {sample['movie']} released in {false_year} describing about?"
        sample["fp_question_3"] = f"What was the {false_year} film {sample['movie']} about?"
        sample["fp_question_4"] = f"Who are the main characters in the {false_year} film {sample['movie']}?"
    return samples

def load_wiki_movie_only_fp(model_size=7,use_docker=False):
    assert model_size in [7,13]
    # base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    base_dir = '/home/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    file = f'{base_dir}/dataset/ToyDataset/Movies/llama2-{model_size}b-chat_on_wiki_movies_0_to_1000_why_fp_question_model_answer.json'
    samples = add_more_fps(read_json(file))
    return samples

if __name__ == '__main__':
    # samples = load_wikidata_movies_knowing(model_size=13)

    samples_13b = load_wiki_movie_only_fp(model_size=13)
    samples_7b = load_wiki_movie_only_fp(model_size=7)


    # samples_original = load_film_release()
    samples = load_wikidata_movies_knowing()
    # num_time = [len(sample["time"]) for sample in samples]
    # print(max(num_time))
    # # remove the model_answer
    # for sample in samples:
    #     del sample["model_answer"]
    # from utils import write_to_json
    # write_to_json(samples,'/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Movies/film_release.json')
