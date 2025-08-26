from utils import read_json, select, write_to_json
import pandas as pd
from scipy.stats import pearsonr


def calc_acc(samples, key='relaxed_result'):
    num_sample = len(samples)
    if num_sample == 0:
        print("Num examples is zero!")
        return 0
    true_count = sum([1 for sample in samples if sample[key]])
    return true_count / num_sample


def print_table(data_dict):
    print(pd.DataFrame.from_dict(data_dict.items()).to_string(index=False, header=False))


def fine_grained_evaluation(samples, valid_digit=3):
    valid_premise_samples = select(samples, false_premise=False)
    fast_valid_premise_samples = select(valid_premise_samples, fact_type='fast-changing')
    slow_valid_premise_samples = select(valid_premise_samples, fact_type='slow-changing')
    never_valid_premise_samples = select(valid_premise_samples, fact_type='never-changing')
    before_2022_valid_premise_samples = select(valid_premise_samples, effective_year='before 2022')
    after_2022_valid_premise_samples = select(valid_premise_samples, effective_year='2022') + select(
        valid_premise_samples, effective_year='2023')
    single_hop_valid_premise_samples = select(samples, num_hops='one-hop')
    multi_hop_valid_premise_samples = select(samples, num_hops='multi-hop')

    false_premise_samples = select(samples, false_premise=True)
    before_2022_false_premise_samples = select(false_premise_samples, effective_year='before 2022')

    # Calculate accuracy for each sample set
    sample_sets = {
        'all': samples,
        'all_valid_premise': valid_premise_samples,
        'fast_valid_premise': fast_valid_premise_samples,
        'slow_valid_premise': slow_valid_premise_samples,
        'never_valid_premise': never_valid_premise_samples,
        'before_2022_valid_premise': before_2022_valid_premise_samples,
        'after_2022_valid_premise': after_2022_valid_premise_samples,
        'single_hop_valid_premise': single_hop_valid_premise_samples,
        'multi_hop_valid_premise': multi_hop_valid_premise_samples,
        'all_false_premise': false_premise_samples,
        'before_2022 false premise': before_2022_false_premise_samples,
    }

    accuracy_results = {}

    for name, samples in sample_sets.items():
        accuracy = calc_acc(samples)
        accuracy_results[name] = round(accuracy, valid_digit)

    return accuracy_results


if __name__ == '__main__':
    print("original prompt:")
    # file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FreshQA/llama2-13b-chat_new_prompt_evaluation_qa_simple_prompt.json'
    # file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FreshQA/llama2-13b-chat_evaluation_qa_simple_prompt.json'
    # file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FreshQA/llama2-13b-chat_evaluation_qa_simple_prompt.json'
    samples = read_json(file)
    original_acc_results = fine_grained_evaluation(samples)
    print_table(original_acc_results)
    #
    # print("--------------")
    # print("qa simple prompt:")
    # file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FreshQA/llama2-7b-chat_evaluation_qa_simple_prompt.json'
    # samples = read_json(file)
    # curr_acc_results = fine_grained_evaluation(samples)
    # print_table(curr_acc_results)
    #
    # print("-----------------------")
    # corr, _ = pearsonr(list(original_acc_results.values()), list(curr_acc_results.values()))
    # print('Pearsons correlation: %.3f' % corr)

    # before_2022_samples = select(samples,effective_year='before 2022')
    # before_2022_acc = calc_acc(before_2022_samples)
    # before_2022_false_premise_samples = select(before_2022_samples,false_premise=True)
    # before_2022_false_premise_acc = calc_acc(before_2022_false_premise_samples)
    # before_2022_false_premise_samples_false = select(before_2022_false_premise_samples,relaxed_result=False)

# set([sample["effective_year"] for sample in samples])


# samples = select(results,false_premise=True)
# relaxed_true_count = sum([1 for sample in samples if sample["relaxed_result"]])
# relaxed_acc = relaxed_true_count/len(samples)
# strict_true_count = sum([1 for sample in samples if sample["strict_result"]])
# strict_acc = strict_true_count / len(samples)
# print(relaxed_acc,strict_acc)
#
# strict_better_than_relaxed = [sample for sample in samples if not sample["relaxed_result"] and sample["strict_result"]]
# strict_better_than_relaxed_idx = [idx for idx,sample in enumerate(samples) if not sample["relaxed_result"] and sample["strict_result"]]
# write_to_json(strict_better_than_relaxed,'/home/zhuoran/hongbang/projects/HalluInducing/experiments/baselines/debug.json')
