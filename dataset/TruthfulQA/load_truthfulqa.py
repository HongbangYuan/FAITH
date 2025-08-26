import json
import csv


def load_truthfulqa(split='generation'):
    assert split in ['generation', 'multiple_choice']
    generation_file = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/TruthfulQA/TruthfulQA.csv'
    multiple_choice_file = '/home/zhuoran/hongbang/projects/HalluInducing/dataset/TruthfulQA/mc_task.json'

    if split == 'multiple_choice':
        # Multiple choice data is in a `JSON` file.
        with open(multiple_choice_file, encoding="utf-8") as f:
            contents = json.load(f)
            samples = []
            for key, row in enumerate(contents):
                samples.append({
                    "question": row["question"],
                    "mc1_targets": {
                        "choices": list(row["mc1_targets"].keys()),
                        "labels": list(row["mc1_targets"].values()),
                    },
                    "mc2_targets": {
                        "choices": list(row["mc2_targets"].keys()),
                        "labels": list(row["mc2_targets"].values()),
                    },
                })

    else:
        # Generation data is in a `CSV` file.
        with open(generation_file, newline="", encoding="utf-8-sig") as f:
            contents = csv.DictReader(f)
            samples = []
            for key, row in enumerate(contents):
                # Ensure that references exist.
                if not row["Correct Answers"] or not row["Incorrect Answers"]:
                    continue
                samples.append({
                    "type": row["Type"],
                    "category": row["Category"],
                    "question": row["Question"],
                    "best_answer": row["Best Answer"],
                    "correct_answers": _split_csv_list(row["Correct Answers"]),
                    "incorrect_answers": _split_csv_list(row["Incorrect Answers"]),
                    "source": row["Source"],
                })
    return samples


def _split_csv_list(csv_list: str, delimiter: str = ";") -> str:
    """
    Splits a csv list field, delimited by `delimiter` (';'), into a list
    of strings.
    """
    csv_list = csv_list.strip().split(delimiter)
    return [item.strip() for item in csv_list]


def load_truthfulqa_repe(user_tag, assistant_tag, preset=""):
    samples = load_truthfulqa(split='multiple_choice')
    questions, answers = [], []
    labels = []
    for d in samples:
        q = d['question']
        for i in range(len(d['mc1_targets']['labels'])):
            a = d['mc1_targets']['choices'][i]
            questions = [f'{user_tag}' + q + ' ' + preset] + questions
            answers = [f'{assistant_tag}' + a] + answers
        ls = d['mc1_targets']['labels']
        ls.reverse()
        labels.insert(0, ls)
    return questions, answers, labels


if __name__ == '__main__':
    # samples = load_truthfulqa(split='multiple_choice')
    samples = load_truthfulqa_repe(user_tag='[INST]', assistant_tag='[/INST]')

