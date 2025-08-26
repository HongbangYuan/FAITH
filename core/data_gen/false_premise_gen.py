from dotenv import load_dotenv

load_dotenv()

import openai
from utils import read_json, select, write_to_json
from core.data_gen.false_premise_gen_agent import FalsePremiseGenAgent

result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FEC/llama2-13b-chat.json'
generated_question_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FEC/llama2-13b-chat_false_premise_questions.json'
results = read_json(result_file)

preds = [True if sample["model_answer"].startswith("Yes") else False for sample in results]
labels = [sample["label"] for sample in results]
for i, sample in enumerate(results):
    sample["pred"] = preds[i]

samples_known = [sample for sample in results if sample["pred"] == sample["label"]]
samples_known_false = select(samples_known, label=False)

agents = []
for sample in samples_known_false:
    agents.append(
        FalsePremiseGenAgent(
            false_fact=sample["input_claim"],
            true_fact=sample["gt_claim"],
            verbose=True,
        )
    )

for idx,(agent,sample) in enumerate(zip(agents,samples_known_false)):
    print("idx:",idx)

    while True:
        try:
            agent.gen()
            break
        except openai.APIConnectionError:
            print("API connection Error! Sleep for 5 seconds and try again...")

    sample["false_premise_question"] = agent.false_premise_question

    print("---------------")
    write_to_json(samples_known_false,generated_question_file)

print("Writing results to {}!".format(generated_question_file))

print("Finished Running!")
