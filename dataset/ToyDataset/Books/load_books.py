import random
import re
random.seed(0)

from utils import read_jsonl,read_json


def filter_samples(samples):
    sample_id_sets = set()
    results = []
    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id not in sample_id_sets:
            sample_id_sets.add(sample_id)
            results.append(sample)
    return results


def load_books(use_docker=False):
    file = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Books/book_author.json'
    if use_docker:
        file = '/mnt/userdata/projects/HalluInducing/dataset/ToyDataset/Books/book_author.json'
    samples = read_jsonl(file)
    filtered_samples = filter_samples(samples)
    print(f"Selected Samples {len(filtered_samples)}/{len(samples)}")

    filtered_samples = add_when_false_premise(add_who_question(filtered_samples))
    return sorted(filtered_samples,key=lambda x: x['subject_pop'])

def load_fp_books(model_size=7,use_docker=False):
    assert model_size == 7 or model_size == 13
    model_size = str(model_size) + 'b'
    file = f"/home/zhuoran/hongbang/projects/HalluInducing/dataset/ToyDataset/Books/llama2-{model_size}-chat_books.json"
    if use_docker:
        file = f'/mnt/userdata/projects/HalluInducing/dataset/ToyDataset/Books/llama2-{model_size}-chat_books.json'
    samples = read_json(file)
    return add_when_fp_2(samples)
    # return samples


def add_who_question(samples):
    for sample in samples:
        sample["who_question"] = f"Who wrote the book {sample['subject_title']}?"
        sample["who_question_ground_truth"] = [sample["object_title"]]
    return samples


def add_when_false_premise(samples):
    author_set = set([sample["object_title"] for sample in samples])
    for sample in samples:
        author = sample["object_title"]
        false_author = random.choice(list(author_set - {author}))
        book_name = sample["subject_title"]
        sample["when_false_premise_question"] = f"When did {false_author} write the book {book_name}?"
    return samples

def add_when_fp_2(samples):
    pattern = r"When did (?P<false_author>.*?) write the book (?P<book_name>.*?)\?"
    for sample in samples:
        question = sample["when_false_premise_question"]
        match = re.match(pattern, question)
        if match:
            false_author = match.group("false_author")
            book_name = match.group("book_name")
        else:
            raise ValueError(f"Author name and book name not found in sentence {sample[question]}")
        sample["when_false_premise_question2"] = f"When was the book {book_name} written by {false_author}?"
    return samples

if __name__ == "__main__":
    # samples = load_fp_books(model_size=13)
    from utils import select
    from core.evaluation.books.books_evaluation import cal_acc_who
    #
    result_file = f'/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/books/llama2-7b-chat_on_books_author_new_fp_when_false_premise_question2_model_answer.json'
    results = read_json(result_file)
    # answer_key = 'when_false_premise_question2_model_answer'
    answer_key = 'when_false_premise_question2_model_answer'
    acc = cal_acc_who(results, key=answer_key)
    print("acc:",acc)
    analyze_samples = select(results, pred=True)
    answers = [sample[answer_key] for sample in analyze_samples]

