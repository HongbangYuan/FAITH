import pandas as pd


def load_oscar_prizes():
    file_path = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Awards/oscars.csv'
    df = pd.read_csv(file_path)

    samples = df.to_dict(orient='records')
    samples = add_yes_no_question(samples)
    samples = add_who_question(samples)
    samples = add_which_question(samples)
    samples = add_why_question_false_premise(samples)

    return samples


def add_who_question(samples):
    for sample in samples:
        name = sample["name"]
        year = sample["oscar_yr"]
        award = sample["award"]
        sample["who_question"] = f"Who won the Academy Award for {award} in {year}?"
        sample["who_question_ground_truth"] = name
    return samples


def add_which_question(samples):
    for sample in samples:
        actor_or_actress = sample["award"].split()[-1].lower()
        name = sample["name"]
        year = sample["oscar_yr"]
        award = sample["award"]
        sample["which_question"] = f"Which {actor_or_actress} win the Academy Award for {award} in {year}?"
        sample["which_question_ground_truth"] = name
    return samples


def add_why_question_false_premise(samples):
    for sample in samples:
        name = sample["name"]
        year = sample["oscar_yr"]
        award = sample["award"]
        sample["why_question_false_premise"] = f"Why {name} failed to win the Academy Award for {award} in {year}?"
        sample["why_long_question_false_premise"] = f"Why {name} failed to win the Academy Award or receive any Academy Award nominations for {award} in {year}?"
    return samples


def add_yes_no_question(samples):
    for sample in samples:
        name = sample["name"]
        year = sample["oscar_yr"]
        award = sample["award"]
        sample["yes_no_question"] = f"Did {name} won the {year} Academy Award for {award}?"
        sample["yes_no_question_wrong_year"] = f"Did {name} won the {year + 1} Academy Award for {award}?"
    return samples


if __name__ == '__main__':
    samples = load_oscar_prizes()
