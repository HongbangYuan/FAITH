# generate false-premise questions according to false questions
from core.data_gen.false_premise_gen_prompts import FALSE_PREMISE_GEN_EXAMPLES,false_premise_gen_prompt
from utils import AnyOpenAILLM
import os

class FalsePremiseGenAgent:
    def __init__(
            self,
            false_fact,
            true_fact,
            verbose=True,
            prompt=false_premise_gen_prompt,
            examples=FALSE_PREMISE_GEN_EXAMPLES,
            llm = AnyOpenAILLM(
                temperature=0,
                max_tokens=250,
                # model_name="gpt-3.5-turbo",
                model_name="gpt-3.5-turbo-1106",
                model_kwargs={"stop": "\n"},
                openai_api_key=os.environ['OPENAI_API_KEY']
            ),
    ):
        self.false_fact = false_fact
        self.true_fact = true_fact
        self.verbose = verbose
        self.prompt = prompt
        self.examples = examples
        self.llm = llm
        self.scratchpad = ''
        self.false_premise_question = None

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def gen(self):
        self.scratchpad += "\n[False Premise Question]"

        false_premise_question = self.prompt_agent()
        self.scratchpad += ' ' + false_premise_question
        self.print(self.scratchpad.split('\n')[-1])

        self.false_premise_question = false_premise_question



    def prompt_agent(self):
        return format_step(self.llm(self._build_prompt()))

    def _build_prompt(self):
        return self.prompt.format(
            examples = self.examples,
            false_fact = self.false_fact,
            true_fact = self.true_fact,
            scratchpad = self.scratchpad
        )


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

