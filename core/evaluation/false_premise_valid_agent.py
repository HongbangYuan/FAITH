from core.evaluation.false_premise_valid_prompt import FALSE_PREMISE_VALID_EXAMPLES, false_premise_valid_prompt
from utils import AnyOpenAILLM
import os


class FalsePremiseValidAgent:
    def __init__(
            self,
            false_fact,
            false_premise_question,
            verbose=True,
            prompt=false_premise_valid_prompt,
            examples=FALSE_PREMISE_VALID_EXAMPLES,
            llm=AnyOpenAILLM(
                temperature=0,
                max_tokens=250,
                model_name="gpt-3.5-turbo",
                model_kwargs={"stop": "\n"},
                openai_api_key=os.environ['OPENAI_API_KEY']
            ),
    ):
        self.false_fact = false_fact
        self.false_premise_qustion = false_premise_question
        self.verbose = verbose
        self.prompt = prompt
        self.examples = examples
        self.llm = llm
        self.scratchpad = ''
        self.is_valid = None

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def check(self):
        # Thought
        self.scratchpad += "\n[Thought]"

        self.scratchpad += ' ' + self.prompt_agent()
        self.print(self.scratchpad.split('\n')[-1])

        # Conclusion
        self.scratchpad += f'\n[Conclusion]'
        conclusion = self.prompt_agent()
        self.scratchpad += ' ' + conclusion
        self.print(self.scratchpad.split('\n')[-1])

        if 'invalid' in conclusion.lower():
            self.is_valid = False
        elif 'valid' in conclusion.lower():
            self.is_valid = True
        else:
            print("Invalid evaluation results. Please try again!")

    def prompt_agent(self):
        return format_step(self.llm(self._build_prompt()))

    def _build_prompt(self):
        return self.prompt.format(
            examples=self.examples,
            false_fact=self.false_fact,
            false_premise_question=self.false_premise_qustion,
            scratchpad=self.scratchpad
        )


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')
