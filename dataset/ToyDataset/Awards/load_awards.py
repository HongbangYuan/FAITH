import pandas as pd
import langid
from collections import defaultdict
import random
random.seed(0)
from utils import read_json

prize_categories = {
    'The Sveriges Riksbank Prize in Economic Sciences in Memory of Alfred Nobel',
    'The Nobel Peace Prize',
    'The Nobel Prize in Physiology or Medicine',
    'The Nobel Prize in  Literature',
    'The Nobel Prize in Chemistry',
    'The Nobel Prize in Physics'
}

def combine_names(row):
    # Combine non-empty elements into a list
    names_list = [row['name'], row['knownName'], row['givenName'], row['familyName'], row['fullName'], row['penName']]
    names_list = [name for name in names_list if pd.notna(name) and name != '']
    return list(set(names_list))

def is_english(text):
    # Use langid to identify the language of the text
    lang, _ = langid.classify(text)

    # Check if the identified language is English
    return lang.lower() in ['en', 'fr', 'de', 'zh']


def load_nobal_prizes():
    file_path = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Awards/nobal_prizes.csv'
    df = pd.read_csv(file_path)

    df['names'] = df.apply(combine_names, axis=1)

    # Drop rows with NaN values in the 'knownName' column
    df = df.dropna(subset=['name'])
    df = df[df['name'].str.lower() != 'nan']
    # Filter rows with English names using langid
    df['is_english'] = df['name'].apply(is_english)
    english_names_df = df[df['is_english']]

    selected_columns = ['awardYear', 'category', 'name', 'birth_country','birth_city','motivation', 'categoryFullName', 'names']
    filtered_data = english_names_df[selected_columns]

    # Convert DataFrame to a list of dictionaries
    result_list = filtered_data.to_dict(orient='records')

    # Sort the list of dictionaries by 'awardYear'
    result_list = sorted(result_list, key=lambda x: x['awardYear'])

    result_list = construct_who(result_list)
    result_list = construct_who_false_premise(result_list)
    result_list = construct_when(result_list)
    result_list = construct_where(result_list)
    result_list = construct_when_fp(result_list)

    return result_list


def choose_other_category(curr_category):
    other_categories = list(prize_categories - {curr_category} - {'The Sveriges Riksbank Prize in Economic Sciences in Memory of Alfred Nobel'})
    return random.choice(other_categories)


def construct_who(samples):
    for sample in samples:
        sample["who_question"] = f"Who won the {sample['categoryFullName']} {sample['motivation']}?"
        sample["who_answer_ground_truth"] = sample["names"]
    return samples

def construct_who_false_premise(samples):
    for sample in samples:
        false_category = choose_other_category(sample["categoryFullName"])
        sample["who_question_false_premise"] = f"Who won the {false_category} {sample['motivation']}?"
    return samples

def construct_when(samples):
    for sample in samples:
        sample["when_question"] = f"When was {sample['name']} be awarded the {sample['categoryFullName']}"
        sample["when_answer_ground_truth"] = sample["awardYear"]
    return samples

def construct_where(samples):
    for sample in samples:
        sample["where_question"] = f"Where was {sample['name']} born?"
        sample["where_question_actually"] = f"Where was {sample['name']} actually born?"
        sample["where_answer_ground_truth"] = [sample["birth_city"],sample["birth_country"]]
    return samples

def construct_when_fp(samples):
    for sample in samples:
        sample["when_fp_question"] = f"For what specific contribution was {sample['name']} awarded {sample['categoryFullName']}in {sample['awardYear']+1}?"
    return samples

def construct_following_up_when_fp_questions(samples):
    for sample in samples:
        false_year = sample['awardYear'] + 1
        sample["when_fp_question2"] = f"{sample['name']} was awarded {sample['categoryFullName']} in {false_year} for what specific reason?"
        sample["when_fp_question3"] = f"{sample['categoryFullName']} in {false_year} was awarded to {sample['name']}  for what specific reason?"
        sample["when_fp_question4"] = f"Why was {sample['name']} awarded the {false_year} {sample['categoryFullName']}?"
    return samples

# def load_who():
#     samples = load_nobal_prizes()
#     samples = construct_who(samples)
#     samples = construct_who_false_premise(samples)
#     return samples


# def load_when_false_repmise():
#     samples = load_nobal_prizes()
#
#     categories = list(set([sample["category"] for sample in samples]))
#
#     # Create a dictionary to store laureate information
#     laureate_dict = defaultdict(list)
#     motivation2laureate = defaultdict(list)
#
#     # Populate the laureate_dict with prize information
#     for entry in samples:
#         laureate_name = entry['name']
#         prize_info = {'awardYear': entry['awardYear'], 'categoryFullName': entry['categoryFullName'],
#                       'motivation': entry['motivation']}
#         laureate_dict[laureate_name].append(prize_info)
#         motivation2laureate[entry["motivation"]].append(laureate_name)
#
#     return laureate_dict, motivation2laureate
    # results = []
    # for sample in samples:
    #     sample["when_false_premise"] = "When did "

def get_id2name(samples):
    results = defaultdict(list)
    for sample in samples:
        sample_id = get_sample_id(sample)
        results[sample_id].extend(sample["who_answer_ground_truth"])
    return results

def get_sample_id(sample):
    return str(sample["awardYear"]) + '_' + sample["category"].replace(" ", "_")


def load_nobel_prize_only_fp(model_size,use_docker=False):
    assert model_size in [7,13]
    base_dir = '/home/zhuoran/hongbang/projects/HalluInducing' if not use_docker else '/mnt/userdata/projects/HalluInducing'
    file = f'{base_dir}/dataset/ToyDataset/Awards/llama2-{model_size}b-chat_on_nobel_prize.json'
    samples = construct_following_up_when_fp_questions(read_json(file))
    return samples


if __name__ == '__main__':
    samples_7b = load_nobel_prize_only_fp(model_size=7)
    samples_13b = load_nobel_prize_only_fp(model_size=13)

    all_samples = set()
    for sample in samples_13b+samples_7b:
        sample_id = str(sample["awardYear"])+sample["category"]+sample["name"]
        if sample_id not in all_samples:
            all_samples.add(sample_id)

    # count = [sample["when_answer_eval"] for sample in samples_13b]

    # samples = load_nobal_prizes()
    # print("Finished!")
    # id2name = get_id2name(samples)

    # samples = load_nobal_prizes()
    # laureate_dict,motivation2laureate = load_when_false_repmise()
    # for key,value in laureate_dict.items():
    #     if len(value) > 1:
    #         print(key)
    #         print(value)
    #         print("-----")
    # for key,value in motivation2laureate.items():
    #     if len(value) > 2:
    #         print(key)
    #         print(value)
    #         print("---------")
