from core.evaluation.fresheval_prompts import relaxed_prompt, strict_prompt, RELAXED_EXAMPLES, STRICT_EXAMPLES, evaluation_qa_prompt
from utils import AnyOpenAILLM
import os


class FreshEvalAgent:
    def __init__(
            self,
            question,
            correct_answer,
            response,
            verbose=True,
            # relaxed_prompt=relaxed_prompt,
            relaxed_prompt=evaluation_qa_prompt,
            strict_prompt=strict_prompt,
            relaxed_examples=RELAXED_EXAMPLES,
            strict_examples=STRICT_EXAMPLES,
            strict_llm=AnyOpenAILLM(
                temperature=0,
                max_tokens=250,
                # model_name="gpt-3.5-turbo",
                model_name="gpt-3.5-turbo-1106",
                model_kwargs={"stop": "\n"},
                openai_api_key=os.environ['OPENAI_API_KEY']
            ),
            relaxed_llm=AnyOpenAILLM(
                temperature=0,
                max_tokens=250,
                # model_name="gpt-3.5-turbo",
                model_name="gpt-3.5-turbo-1106",
                model_kwargs={"stop": "\n"},
                openai_api_key=os.environ['OPENAI_API_KEY']
            )
    ):
        self.question = question
        self.correct_answer = correct_answer
        self.response = response
        self.verbose = verbose
        self.relaxed_prompt = relaxed_prompt
        self.strict_prompt = strict_prompt
        self.relaxed_examples = relaxed_examples
        self.strict_examples = strict_examples
        self.strict_llm = strict_llm
        self.relaxed_llm = relaxed_llm
        self.relaxed_scratchpad = ''
        self.strict_scratchpad = ''
        self.relaxed_result = False
        self.strict_result = False

    def _check_mode_arg(self,mode):
        assert mode in ["relaxed","strict"]

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def strict_step(self):
        # comment
        self.strict_scratchpad += f'\n[Comment]:'
        self.strict_scratchpad += ' ' + self.prompt_agent(mode='strict')
        self.print(self.strict_scratchpad.split('\n')[-1])

        # evaluation
        self.strict_scratchpad += f'\n[Evaluation]:'
        evaluation = self.prompt_agent(mode="strict")
        self.strict_scratchpad += ' ' + evaluation
        self.print(self.strict_scratchpad.split('\n')[-1])

        if 'incorrect' in evaluation.lower():
            self.strict_result = False
        elif 'correct' in evaluation.lower():
            self.strict_result = True
        else:
            # self.print("Invalid evaluation results. Please try again.")
            print("Invalid evaluation results. Please try again.")

    def relaxed_step(self):
        # comment
        self.relaxed_scratchpad += f'\n[Comment]:'
        self.relaxed_scratchpad += ' ' + self.prompt_agent(mode='relaxed')
        self.print(self.relaxed_scratchpad.split('\n')[-1])

        # evaluation
        self.relaxed_scratchpad += f'\n[Evaluation]:'
        evaluation = self.prompt_agent(mode="relaxed")
        self.relaxed_scratchpad += ' ' + evaluation
        self.print(self.relaxed_scratchpad.split('\n')[-1])

        if 'incorrect' in evaluation.lower():
            self.relaxed_result = False
        elif 'correct' in evaluation.lower():
            self.relaxed_result = True
        else:
            # self.print("Invalid evaluation results. Please try again.")
            print("Invalid evaluation results. Please try again.")

    def prompt_agent(self, mode) -> str:
        return format_step(self.relaxed_llm(self._build_prompt(mode)))

    def _build_prompt(self,mode):
        self._check_mode_arg(mode)
        if mode == 'relaxed':
            return self._build_relaxed_prompt()
        else:
            return self._build_strict_prompt()

    def _build_relaxed_prompt(self) -> str:
        return self.relaxed_prompt.format(
            examples=self.relaxed_examples,
            question=self.question,
            correct_answer=self.correct_answer,
            response=self.response,
            scratchpad=self.relaxed_scratchpad
        )

    def _build_strict_prompt(self) -> str:
        return self.strict_prompt.format(
            examples=self.strict_examples,
            question=self.question,
            correct_answer=self.correct_answer,
            response=self.response,
            scratchpad=self.strict_scratchpad
        )


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')
