from utils import read_jsonl
import os


def load_invite():
    dataset_path = '/home/zhuoran/hongbang/projects/invite-llm-hallucinations/data_dir/dbpedia/predicate_wise_questions'
    result_dict = {}
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        samples = read_jsonl(file_path)
        result_dict[file] = samples
    return result_dict


if __name__ == '__main__':
    result_dict = load_invite()

