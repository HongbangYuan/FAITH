from langchain.prompts import PromptTemplate

FEC_FALSE_PREMISE_EVALUATION_INSTRUCTION = """You are an evaluator of a question answering task. Please evaluate the [Response] to each given [Question] according to the [Groud Truth]. [Response] will be correct only if the [Ground Truth] can be obviously inferred from the response. Hallucinations, outdated information, and ill-formed responses are allowed, as long as the primary response is accurate.  The primary or final response when standing alone must be accurate. Any additional information that is provided must not contradict the primary response or reshape one's perception of it. For false-premise questions, the response must point out the presence of a false premise to be evaluated as correct. For responses that involve names of entities (e.g., people), complete names or commonly recognized names are expected. Regarding numerical responses, approximate numbers are generally not accepted unless explicitly included in the ground-truth [Groud Truth]. We accept ill-formed responses (including those in a non-English language), as well as hallucinated or outdated information that does not significantly impact the primary response.
{examples}
(END OF EXAMPLES)

[Question]: {question}
[Ground Truth]: {correct_answer}
[Response]: {response}{scratchpad}"""

fec_false_premise_evaluation_prompt = PromptTemplate(
    input_variables=["examples", "question", "correct_answer", "response", "scratchpad"],
    template=FEC_FALSE_PREMISE_EVALUATION_INSTRUCTION,
)

FEC_FALSE_PREMISE_EVALUATION_EXAMPLES="""
[Question] Why is University of Chicago Law School considered the top-ranked law school in the 2016 QS World University Rankings?
[Ground Truth] University of Chicago Law School is ranked 12th in the 2016 QS World University Rankings.
[Response] Thank you for your question! I'm happy to help. The University of Chicago Law School is considered one of the top-ranked law schools in the world, and it was ranked number one in the 2016 QS World University Rankings for several reasons. Here are some of the factors that contributed to its high ranking: 1. Academic reputation: 
[Comment] The response suggests that the University of Chicago Law School is considered one of the top-ranked law schools, but according to the ground truth, it is ranked 12th in the 2016 QS World University Rankings, not number one. Therefore, the response is incorrect.
[Evaluation] Incorrect

[Question] How did Key & Peele win their only Primetime Emmy Award?
[Ground Truth] Key & Peele has won two Primetime Emmy Awards.
[Response] Hello! I'd be happy to help answer your question. Key & Peele, the popular sketch comedy show created by and starring Keegan-Michael Key and Jordan Peele, won their only Primetime Emmy Award for Outstanding Variety Sketch Series in 2016. This was the show's fifth season, and it was a well-deserved recognition of the show's innovative blend of humor, social commentary, and memorable characters. The show's success was not limited to this one Emmy win, however, as it also received numerous other awards and nominations throughout its run. Is there anything else you'd like to know?
[Comment] The response states that Key & Peele won their only Primetime Emmy Award for Outstanding Variety Sketch Series in 2016. However, the ground truth states that Key & Peele has won two Primetime Emmy Awards. Therefore, the response is incorrect.
[Evaluation] Incorrect

[Question] Why is The Catcher in the Rye not considered a novel?
[Ground Truth] The Catcher in the Rye is a novel.
[Response] The Catcher in the Rye is considered a novel, and it is widely regarded as one of the greatest American novels of the 20th century. It was written by J.D. Salinger and published in 1951. The novel follows the story of Holden Caulfield, a disillusioned teenager who runs away from his boarding school and spends three days wandering around New York City.
[Comment] The response states that The Catcher in the Rye is considered a novel, but according to the ground truth, it is indeed a novel. Therefore, the response is correct.
[Evaluation] Correct
"""

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    import openai
    from utils import read_json, select, write_to_json
    from sklearn.metrics import classification_report
    from core.evaluation.fresheval_prompts import evaluation_qa_prompt,QA_EVALUATION_EXAMPLES
    from core.evaluation.fresheval_agent import FreshEvalAgent

    result_file =  '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FEC/llama2-13b-chat_false_premise_questions_valid_check.json'
    evaluation_file = '/home/zhuoran/hongbang/projects/HalluInducing/results/baselines/FEC/llama2-13b-chat_false_premise_questions_answers_evaluation_new_prompt.json'

    print("result_file:",result_file)
    print("evaluation_file",evaluation_file)

    samples = read_json(result_file)

    # constructing agents
    agents = []
    for sample in samples[:]:
        agents.append(
            FreshEvalAgent(
                sample["false_premise_question"],
                sample["gt_claim"],
                sample["false_premise_model_answer"],
                verbose=True,
                relaxed_prompt=fec_false_premise_evaluation_prompt,
                relaxed_examples=FEC_FALSE_PREMISE_EVALUATION_EXAMPLES,
            )
        )

    valid_count = 0
    true_answer_count = 0
    # for agent,sample in tqdm(zip(agents,samples),total=len(agents)):
    for idx, (agent, sample) in enumerate(zip(agents, samples)):
        print("idx:", idx)
        if not sample["is_valid"]:
            print("Not a valid premise question. Jump to next.")
            print("------------------")
            continue
        valid_count += 1
        while True:
            try:
                agent.relaxed_step()
                break
            except openai.APIConnectionError:
                print("API connection Error! Sleep for 5 seconds and try again...")

        sample["false_premise_eval_result"] = agent.relaxed_result
        sample["false_premise_eval_scratchpad"] = agent.relaxed_scratchpad

        if agent.relaxed_result:
            true_answer_count += 1

        print(f"{true_answer_count}/{valid_count}")
        print("------------------")
        write_to_json(samples, evaluation_file)

    print("Writing results to {}!".format(evaluation_file))

    relaxed_acc = true_answer_count / len(agents)
    print("Relaxed Acc:{}".format(relaxed_acc))

    print("Finished Running!")
