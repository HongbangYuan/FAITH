# yes no question evalation
from utils import read_json,select
from sklearn.metrics import classification_report

result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/QAQA/llama2-13b-chat_yesno_verification_question_including_valid_qs_new_format.json'
# result_file = '/home/zhuoran/hongbang/projects/HalluInducing/experiments/baselines/debug_humaneval.json'
results = read_json(result_file)

a = [sample["model_answer"] for sample in select(results)]

examples = [sample for sample in results if not "yes" in sample and not "No," in sample]
preds = [True if 'yes' in sample["model_answer"] or 'Yes' in sample["model_answer"] else False for sample in results ]
labels = [True if sample["all_assumptions_valid"] == 'all_valid' else False for sample in results]
print(classification_report(labels,preds))
