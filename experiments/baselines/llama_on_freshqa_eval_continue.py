import openai
from dotenv import load_dotenv

load_dotenv()

from utils import read_json, write_to_pickle, write_to_json
from core.evaluation.fresheval_agent import FreshEvalAgent
from tqdm import tqdm
from core.evaluation.fresheval_prompts import evaluation_qa_prompt,QA_EVALUATION_EXAMPLES


evaluation_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FreshQA/llama2-13b-chat_new_prompt_evaluation_qa_simple_prompt.json'

samples = read_json(evaluation_file)


# constructing agents
agents = []
for sample in samples[:]:
    agents.append(
        FreshEvalAgent(
            sample["question"],
            "\n".join([str(answer) for answer in sample["answers"]]),
            sample["model_answer"],
            verbose=True,
            relaxed_prompt=evaluation_qa_prompt,
            relaxed_examples=QA_EVALUATION_EXAMPLES,
        )
    )

strict_true_count = 0
relaxed_true_count = 0
# for agent,sample in tqdm(zip(agents,samples),total=len(agents)):
for idx,(agent,sample) in enumerate(zip(agents,samples)):
    if "relaxed_result" in sample and "relaxed_scratchpad" in sample:
        if sample["relaxed_result"]:
            relaxed_true_count += 1
        continue
    print("idx:",idx)
    while True:
        try:
            agent.relaxed_step()
            break
        except openai.APIConnectionError:
            print("API connection Error! Sleep for 5 seconds and try again...")

    sample["relaxed_result"] = agent.relaxed_result
    sample["relaxed_scratchpad"] = agent.relaxed_scratchpad

    if agent.relaxed_result:
        relaxed_true_count += 1

    print("------------------")
    write_to_json(samples,evaluation_file)


print("Writing results to {}!".format(evaluation_file))

relaxed_acc = relaxed_true_count / len(agents)
print("Relaxed Acc:{}".format(relaxed_acc))

print("Finished Running!")
