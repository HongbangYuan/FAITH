from utils import read_json,select
from sklearn.metrics import classification_report

result_file =  '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FEC/llama2-13b-chat.json'
# result_file = '/home/zhuoran/hongbang/projects/HalluInducing/experiments/baselines/debug_humaneval.json'
results = read_json(result_file)

preds = [True if sample["model_answer"].startswith("Yes") else False for sample in results]
labels = [sample["label"] for sample in results]
for i,sample in enumerate(results):
    sample["pred"] = preds[i]

samples_known = [sample for sample in results if sample["pred"] == sample["label"]]
samples_known_false = select(samples_known,label=False)

print(classification_report(labels,preds))
