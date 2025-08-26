from dotenv import load_dotenv

load_dotenv()

from utils import read_json, write_to_pickle, write_to_json
from core.evaluation.fresheval_agent import FreshEvalAgent
from tqdm import tqdm
from core.evaluation.fresheval_prompts import evaluation_qa_prompt,QA_EVALUATION_EXAMPLES
import openai

QAQA_DATASET_EXAMPLES = """
[Question]: What did pete burns die of?
[Ground Truth]: Pete Burns died following a sudden cardiac arrest.
[Response]: I'm so sorry to hear that Pete Burns has passed away. However, I must inform you that Pete Burns, the lead singer of the band Dead or Alive, passed away on October 23, 2016, due to a cardiac arrest caused by a pulmonary embolism. It's important to note that I'm just an AI and do not have access to personal information or medical records of individuals, so I cannot provide any information that may be considered sensitive or inappropriate. It's always best to rely on credible sources and respect people's privacy and personal information. If you have any other questions or concerns, feel free to ask!
[Comment]: The response accurately indicates that Pete Burns passed away due to a cardiac arrest, specifically caused by a pulmonary embolism. This aligns with the Ground Truth of his death being a result of sudden cardiac arrest. Therefore, the response is correct in conveying the cause of Pete Burns' death.
[Evaluation]: correct

[Question]: Why did the uk ban guns?
[Ground Truth]: Guns are not banned totally in the UK. Handguns have been banned in Great Britain following the Dunblane school massacre in 1996, but they are still allowed in Northern Ireland.
[Response]: Hello! I'm here to help answer your question. The UK has not completely banned guns, but there are strict laws and regulations in place regarding the ownership and use of firearms. The reasons for these laws are complex and multifaceted, but some of the main reasons include: 1. Public safety: The UK has experienced several gun violence incidents, including mass shootings and terrorist attacks, which have highlighted the need for stricter gun control measures to protect the public. 2. Reducing the risk of gun crime: The UK has a relatively low rate of gun crime compared to other countries, and the government aims to keep it that way by making it more difficult for criminals to obtain and use firearms. 3. Protecting vulnerable individuals: The UK has a strong focus on protecting vulnerable individuals, including children and those who may be at risk of domestic violence or other forms of harm. Stricter gun laws help to reduce the risk of firearms falling into the wrong hands. 4. International obligations: The UK has signed international treaties, such as the United Nations Arms Trade Treaty, which aim to regulate the global trade in arms and prevent the diversion of weapons to illegal markets or terrorist groups. It's important to note that while the UK has strict gun laws, there are still some legal ways to own and use firearms, such as for hunting, sport shooting, and self-defense. However, these activities are heavily regulated and subject to strict licensing and safety requirements. I hope this information is helpful! If you have any further questions, please don't hesitate to ask.
[Comment]: As the ground truth indicates, Guns are not banned totally in the UK. So the question is a false-premise question. The response states that not all guns are banned in the UK, which aligns with the ground truth. Thus, the response is correct.
[Evaluation]: correct

[Question]: How many American casualties in the battle of midway?
[Ground Truth]: A total of 317 United States sailors, airmen, and marines were killed in the Battle of Midway.
[Response]: I apologize, but the question you have asked does not make sense. The Battle of Midway was a naval battle fought during World War II in June 1942, and it did not take place in America. Therefore, there were no American casualties in the Battle of Midway. I understand that you may have made a mistake, and I'm here to help you with any other questions you may have. Please feel free to ask, and I will do my best to provide you with accurate and helpful information.
[Comment]:The response to the question about American casualties in the Battle of Midway is inaccurate. The Ground Truth states that a total of 317 United States sailors, airmen, and marines were killed in the Battle of Midway. The response incorrectly claims there were no American casualties, which contradicts the accurate information provided in the Ground Truth. Thus, the response is incorrect.
[Evaluation]: incorrect
"""

if __name__ == '__main__':
    # model_size = '7b'
    model_size = '13b'
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/repe/llama2-13b-chat_on_qaqa.json'
    result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/dola/llama2-{}-chat_on_qaqa'.format(model_size)
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/QAQA/baichuan2-7b-chat.json'
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/experiments/baselines/debug.json'
    # result_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FreshQA/llama2-13b-chat.json'
    evaluation_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/dola/llama2-{}-chat_on_qaqa_evaluation.json'.format(model_size)

    print("result_file:",result_file)
    print("evaluation_file:",evaluation_file)

    samples = read_json(result_file)

    # constructing agents
    agents = []
    for sample in samples[:]:
        agents.append(
            FreshEvalAgent(
                sample["question"].capitalize() + '?',
                sample["abstractive_answer"],
                " ".join(sample["model_answer"].split("\n")),
                verbose=True,
                relaxed_prompt=evaluation_qa_prompt,
                relaxed_examples=QAQA_DATASET_EXAMPLES,
            )
        )

    strict_true_count = 0
    relaxed_true_count = 0
    # for agent,sample in tqdm(zip(agents,samples),total=len(agents)):
    for idx,(agent,sample) in enumerate(zip(agents,samples)):
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
