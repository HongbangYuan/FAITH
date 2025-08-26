from langchain.prompts import PromptTemplate

RELAXED_PROMPT_INSTRUCTION = """Please evaluate the [Response] to each given [Question] referring to the [Correct Answer(s)]  under a relaxed evaluation, where hallucinations, outdated information, and ill-formed responses are allowed, as long as the primary response is accurate. Please credit the response only if it provides a confident and definitive response, or the [Correct Answer(s)] can be obviously inferred from the response. The primary or final response when standing alone must be accurate. Any additional information that is provided must not contradict the primary response or reshape one's perception of it. For false-premise questions, the response must point out the presence of a false premise to receive credit. For responses that involve names of entities (e.g., people), complete names or commonly recognized names are expected. Regarding numerical responses, approximate numbers are generally not accepted unless explicitly included in the ground-truth [Correct Answer(s)]. We accept ill-formed responses (including those in a non-English language), as well as hallucinated or outdated information that does not significantly impact the primary response.
# some demonstrations are omitted for brevity
{examples}
(END OF EXAMPLES)

Question: {question}
Correct Answer(s): {correct_answer}
Response: {response}{scratchpad}"""

SIMPLE_QA_INSTRUCTION = """You are an evaluator of a question answering task. Please evaluate the [Response] to each given [Question] according to the [Groud Truth]. [Response] will be correct only if the [Ground Truth] can be obviously inferred from the response. Hallucinations, outdated information, and ill-formed responses are allowed, as long as the primary response is accurate.  The primary or final response when standing alone must be accurate. Any additional information that is provided must not contradict the primary response or reshape one's perception of it. For false-premise questions, the response must point out the presence of a false premise to be evaluated as correct. For responses that involve names of entities (e.g., people), complete names or commonly recognized names are expected. Regarding numerical responses, approximate numbers are generally not accepted unless explicitly included in the ground-truth [Groud Truth]. We accept ill-formed responses (including those in a non-English language), as well as hallucinated or outdated information that does not significantly impact the primary response.
{examples}
(END OF EXAMPLES)

[Question]: {question}
[Ground Truth]: {correct_answer}
[Response]: {response}{scratchpad}"""

STRICT_PROMPT_INSTRUCTION = """Please evaluate the [Response] to each given [Question] referring to the [Correct Answer(s)] under a strict evaluation, where no hallucinations, outdated information, or ill-formed responses are allowed. Please credit the response only if it provides a confident and definitive response, or the [Correct Answer(s)] can be obviously inferred from the response. The primary or final response when standing alone must be accurate. Any additional information that is provided must not contradict the primary response or reshape one's perception of it. For false-premise questions, the response must point out the presence of a false premise to receive credit. For responses that involve names of entities (e.g., people), complete names or commonly recognized names are expected. Regarding numerical answers, approximate numbers are generally not accepted unless explicitly included in the ground-truth [Correct Answer(s)]. A response that contains any hallucination, no matter how minor, will not receive credit. Furthermore, when the response indicates that the information might be outdated, we accept it only if it is evident that the knowledge has not changed (e.g., through common sense or well-known facts).
# some demonstrations are omitted for brevity
{examples}
(END OF EXAMPLES)

Question: {question}
Correct Answer(s): {correct_answer}
Response: {response}{scratchpad}"""

RELAXED_EXAMPLES="""
Question: Is Tesla's stock price above $250 a share?
correct answer(s): Yes
Response: Yes, it is. The stock price is currently at $207.
Comment: This is a valid question. While the primary answer in the response (Yes) is accurate, the additional
information contradicts the primary answer ($207 is not above $250). Thus, the response is not credited.
Evaluation: incorrect

Question: Who is the latest winner of the Formula 1 world championship?
Correct answer(s): Max Verstappen | Max Emilian Verstappen
Response: Max Verstappen won the 2022 Formula 1 world championship. He is a Dutch racing driver who
competes in Formula One for Red Bull Racing. He is the son of former Formula One driver Jos Verstappen.
Verstappen started his racing career in karting at the age of seven. He won the 2013 European Karting
Championship and the 2013 World Karting Championship. In 2014, he moved to single-seater racing, competing
in the European Formula Three Championship. He won the championship in his first season, becoming the first
driver to do so since Nico Rosberg in 2005… # the rest is omitted for brevity
Comment: This is a valid question. Even though the response contains several instances of hallucinated
information (e.g., Max Verstappen did not win the Formula Three European Championship in 2014), the primary
answer in the response (Max Verstappen) is still accurate. Thus, the response is credited.
Evaluation: correct
"""


QA_EVALUATION_EXAMPLES="""
[Question]: Is Tesla's stock price above $250 a share?
[Ground Truth]: Yes
[Response]: Yes, it is. The stock price is currently at $207.
[Comment]: While the primary answer in the response (Yes) is accurate, the additional
information contradicts the question ($207 is not above $250). Thus, the question is valid but the response is incorrect. 
[Evaluation]: incorrect

[Question]: When did the population of United States reach 8 billion?
[Ground Truth]: The population of United States has not yet reached 8 billion.
[Response]: The population of the United States reached 8 billion in 2022.
[Comment]: As the ground truth indicates, the polulation of the United States has not yet reached 8 billion. So the question is a false-premise question. The response states that 
the population of the United States reached 8 billion in 2022, which contradicts the ground truth. Thus, the response is incorrect.
[Evaluation]: incorrect

[Question]: Who is the latest winner of the Formula 1 world championship?
[Ground Truth]: Max Verstappen | Max Emilian Verstappen
[Response]: Max Verstappen won the 2022 Formula 1 world championship. He is a Dutch racing driver who
competes in Formula One for Red Bull Racing. He is the son of former Formula One driver Jos Verstappen.
Verstappen started his racing career in karting at the age of seven. He won the 2013 European Karting
Championship and the 2013 World Karting Championship. In 2014, he moved to single-seater racing, competing
in the European Formula Three Championship. He won the championship in his first season, becoming the first
driver to do so since Nico Rosberg in 2005… # the rest is omitted for brevity
[Comment]:  Even though the response contains several instances of hallucinated
information (e.g., Max Verstappen did not win the Formula Three European Championship in 2014), the primary
part in the response (Max Verstappen) is still accurate. This is a valid question and the response is correct.
[Evaluation]: correct
"""

STRICT_EXAMPLES="""
Question: How old is the world's oldest verified living person?
Correct answer(s): 116 years old | 116
Response: As of today {current_date} , the most up-to-date and relevant information regarding this query is as
follows. The oldest verified living person is Maria Branyas Morera, who was born on March 4, 1907, making her 116
years old.
Comment: This is a valid question. The primary answer in the response (116 years old) is accurate, and all the
provided information is accurate and up-to-date. Thus, the response is credited.
Evaluation: correct

Question: Who is the latest winner of the Formula 1 world championship?
correct answer(s): Max Verstappen | Max Emilian Verstappen
Response: Max Verstappen won the 2022 Formula 1 world championship. He is a Dutch racing driver who
competes in Formula One for Red Bull Racing. He is the son of former Formula One driver Jos Verstappen.
Verstappen started his racing career in karting at the age of seven. He won the 2013 European Karting
Championship and the 2013 World Karting Championship. In 2014, he moved to single-seater racing, competing in
the European Formula Three Championship. He won the championship in his first season, becoming the first driver
to do so since Nico Rosberg in 2005… # the rest is omitted for brevity
Comment: This is a valid question. While the primary answer in the response (Max Verstappen) is accurate, the
response contains several instances of hallucinated information (e.g., Max Verstappen did not win the Formula
Three European Championship in 2014). Thus, the response is not credited.
Evaluation: incorrect
"""

evaluation_qa_prompt = PromptTemplate(
    input_variables=["examples","question", "correct_answer", "response","scratchpad"],
    template=SIMPLE_QA_INSTRUCTION,
)

relaxed_prompt = PromptTemplate(
    input_variables=["examples","question", "correct_answer", "response","scratchpad"],
    template=RELAXED_PROMPT_INSTRUCTION,
)

strict_prompt = PromptTemplate(
    input_variables=["examples","question", "correct_answer", "response","scratchpad"],
    template=STRICT_PROMPT_INSTRUCTION,
)

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    from utils import read_json
    from core.evaluation.fresheval_agent import FreshEvalAgent

    result_file = '/home/zhuoran/hongbang/projects/HalluInducing/experiments/baselines/debug.json'


    samples = read_json(result_file)

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

    agent = agents[6]
    relaxed_prompt = agent._build_relaxed_prompt()
