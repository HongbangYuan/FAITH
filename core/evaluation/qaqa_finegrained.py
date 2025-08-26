import pandas as pd
from utils import select
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def calc_acc(samples, key='relaxed_result'):
    num_sample = len(samples)
    if num_sample == 0:
        print("Num examples is zero!")
        return 0
    true_count = sum([1 for sample in samples if sample[key]])
    return true_count / num_sample

def print_table(data_dict):
    print(pd.DataFrame.from_dict(data_dict.items()).to_string(index=False, header=False))

def fine_grained_evaluation(samples,valid_digit=3):
    valid_premise_samples = select(samples,all_assumptions_valid='all_valid')
    false_premise_samples = select(samples,all_assumptions_valid='has_invalid')

    # Calculate accuracy for each sample set
    sample_sets = {
        'all': samples,
        'valid_premise': valid_premise_samples,
        'false_premise': false_premise_samples,
    }

    accuracy_results = {}

    for name, samples in sample_sets.items():
        accuracy = calc_acc(samples)
        accuracy_results[name] = round(accuracy, valid_digit)

    return accuracy_results

def yes_no_evaluation(samples):
    preds = [True if 'yes' in sample["model_answer"] or 'Yes' in sample["model_answer"] else False for sample in samples]
    labels = [True if sample["all_assumptions_valid"] == 'all_valid' else False for sample in samples]
    return classification_report(labels,preds),preds,labels

def cal_acc_yes_no(samples,key,output_key='pred'):
    true_count = 0
    for sample in samples:
        pred = True if 'yes' in sample[key].lower() else False
        label = True if sample["all_assumptions_valid"] == 'all_valid' else False
        if pred == label:
            true_count += 1
        sample[output_key] = (pred == label)
    return true_count / len(samples)

if __name__ == '__main__':
    from utils import read_json

    result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/QAQA/llama2-13b-chat_new_format_evaluation.json'
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/dola/llama2-7b-chat_on_qaqa_evaluation.json'
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/repe/llama2-13b-chat_on_qaqa_evaluation.json'
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/QAQA/baichuan2-13b-chat_evaluation.json'
    samples = read_json(result_file)
    acc_results = fine_grained_evaluation(samples)
    print_table(acc_results)
    fp_samples = select(samples,all_assumptions_valid='has_invalid')
    answers = [fp_sample["model_answer"] for fp_sample in fp_samples]

    yes_no_result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/QAQA/llama2-7b-chat_yesno_verification_question_including_valid_qs_new_format.json'
    results = read_json(yes_no_result_file)
    report,preds,labels = yes_no_evaluation(results)
    print(report)

    # plot the confusion matrix
    conf_matrix = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.title(f"Llama2-7b-chat F1=0.64")
    plt.show()

    plt.imshow(conf_matrix,cmap='Blues')
    plt.show()

    for i,sample in enumerate(samples):
        sample["pred"] = preds[i]
        sample["label"] = labels[i]
        sample["yes_no_prediction"] = results[i]["model_answer"]
        sample["idx"] = i

    false_premise_samples = select(samples,all_assumptions_valid="has_invalid",relaxed_result=False)

    # pick samples that answer the false-premise wrong but got the yes-no question right
    samples_to_pick = []
    for i,sample in enumerate(samples):
        if sample["all_assumptions_valid"] == 'has_invalid':
            if not sample["relaxed_result"]:
                pred = preds[i]
                label = labels[i]
                # if pred == label:
                samples_to_pick.append(sample)




