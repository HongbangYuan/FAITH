from langchain.prompts import PromptTemplate

FALSE_PREMISE_GEN_INSTRUCTION = """You will be presented a [False Fact] and a corresponding [True Fact]. Compare these two facts and generate a coherent and natural false premise questions based on the [False Fact]. The question should be tricky and deceptive as it indicates a false premise or a false presupposition. Remember to incorporate or indicate the false information in the question. For example:
{examples}
(END OF EXAMPLES)

[False Fact]: {false_fact}
[True Fact]: {true_fact}{scratchpad}"""

FALSE_PREMISE_GEN_EXAMPLES = """
[False Fact] Night of the Living Dead is a Spanish comic book.
[True Fact] Night of the Living Dead is a 1990 American horror film.
[False Premise Question] What the Spanish comic book Night of the Living Dead is about?

[False Fact]  Watertown, Massachusetts is in France.
[True Fact] Watertown, Massachusetts is in Massachusetts.
[False Premise Question] Why does Watertown, Massachusetts locate in Frace?

[False Fact] LinkedIn is available in zero languages as of 2013.
[True Fact] LinkedIn is available in 24 languages as of 2013.
[False Premise Question] Why LinkedIn has no available language as of 2013?

[False Fact] Simon Pegg is only a banker.
[True Fact] Simon Pegg is an actor.
[False Premise Question] What does the banker Simon Pegg famous for?
"""

false_premise_gen_prompt = PromptTemplate(
    input_variables=["examples","false_fact","true_fact","scratchpad"],
    template=FALSE_PREMISE_GEN_INSTRUCTION,
)
