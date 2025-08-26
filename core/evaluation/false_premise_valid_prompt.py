from langchain.prompts import PromptTemplate

FALSE_PREMISE_VALID_INSTRUCTION="""You are responsible for checking the validation of the [False Premise Question] according to the given [False Fact] by first having a [Thought], then coming to a [Conclusion] (Valid or Invalid). [False Premise Question] will be valid only if the [False Fact] can be obviously inferred from the [False Premise Question]. 
{examples}
(END OF EXAMPLES)

[False Fact]: {false_fact}
[False Premise Question]: {false_premise_question}{scratchpad}"""

FALSE_PREMISE_VALID_EXAMPLES="""
[False Fact]: Night of the Living Dead is a Spanish comic book.
[False Premise Question]: What are the most popular Spanish comic books of all time?
[Thought] The false premise question is asking for the most popular Spanish comic books of all time. However, the false fact states that Night of the Living Dead is a Spanish comic book. Based on this information, it is clear that the false fact (Night of the Living Dead being a Spanish comic book) cannot be obviously inferred from the false premise question. Therefore, the false premise question is invalid.
[Conclusion] Invalid

[False Fact] Doug Petrie has yet to appear on the web.
[False Premise Question] Why hasn\'t Doug Petrie made any appearances on the web yet?
[Thought] The false premise question is asking why Doug Petrie hasn't made any appearances on the web yet. The false fact states that Doug Petrie has yet to appear on the web. Based on this information, it can be inferred that the false fact (Doug Petrie not appearing on the web) is obviously implied by the false premise question. Therefore, the false premise question is valid.
[Conclusion] Valid
"""

false_premise_valid_prompt = PromptTemplate(
    input_variables=["examples","false_fact", "false_premise_question","scratchpad"],
    template=FALSE_PREMISE_VALID_INSTRUCTION,
)
