from dotenv import load_dotenv

load_dotenv()

import openai
from utils import read_json, select, write_to_json
from core.evaluation.false_premise_valid_agent import FalsePremiseValidAgent


generated_question_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FEC/llama2-13b-chat_false_premise_questions_answers.json'
question_validation_result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FEC/llama2-13b-chat_false_premise_questions_valid_check.json'
samples = read_json(generated_question_file)


agents = []
for sample in samples:
    agents.append(
        FalsePremiseValidAgent(
            false_fact=sample["input_claim"],
            false_premise_question=sample["false_premise_question"],
            verbose=True,
        )
    )

valid_count = 0
num_sample = len(samples)
for idx,(agent,sample) in enumerate(zip(agents,samples)):
    print(f"idx:{idx}")

    while True:
        try:
            agent.check()
            break
        except openai.APIConnectionError:
            print("API connection Error! Sleep for 5 seconds and try again...")

    sample["is_valid"] = agent.is_valid
    sample["is_valid_scratchpad"] = agent.scratchpad

    if agent.is_valid:
        valid_count += 1

    print(f"Valid_count:{valid_count}/{idx+1}  {valid_count/(idx+1)}")
    print("---------------")
    write_to_json(samples,question_validation_result_file)

print("Writing results to {}!".format(generated_question_file))
print("Finished Running!")
