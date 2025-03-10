from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent

from moe.base.expert import BaseExpert
from moe.prompts.prompt_repo import PromptRepo

from langchain_core.runnables import RunnableLambda

from dev_tools.enums.llms import LLMs
from dev_tools.enums.prompt_parsers import PromptParsers


class Router(BaseExpert):
    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or Router.__name__,
            description=description or Router.__doc__,
            strategy=strategy,
            agent=agent or RunnableLambda(lambda state: (
                f"\nAction: {IntentXtractor.__name__}"
                f"\nAction Input: {state['input']}"
            )),
        )


class IntentXtractor(BaseExpert):
    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or IntentXtractor.__name__,
            description=description or IntentXtractor.__doc__,
            strategy=strategy,
            agent=EphemeralNLPAgent(
                name='IntentXtractionAgent',
                llm=llm,
                system_prompt=(
                    '## Objective\n'
                    'You are an advanced reasoning expert tasked with generating a concise and insightful synthesis of a given input query. '
                    'Your goal is to remove noise from the input, clearly exposing the underlying goal that is tried to achieve.\n'
                    '**IMPORTANT:** Do not answer the query directly, nor ask for clarifications!!'
                ),
                prompt_template=(
                    '## Instructions\n'
                    '**Generate a brief assessment that expresses clearly what is being tried to achieve:**\n\n'
                    '{input}'
                )
            ),
        )


class PlanningXpert(BaseExpert):
    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or PlanningXpert.__name__,
            description=description or PlanningXpert.__doc__,
            strategy=strategy,
            agent=EphemeralNLPAgent(
                name='PlanningAgent',
                llm=llm.bind(stop=["\nExpert Response:"]),
                prompt_template=PromptRepo.MoE_ReAct(),
                system_prompt=(
                    '## System Information\n'
                    'You are a strategic decision-making agent who excels at immediate action planning. '
                    'Keep your plan concise.\n'
                    # 'Consider the following chat history (if any).'
                ),
            ),
        )


class SynthesisXpert(BaseExpert):
    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or SynthesisXpert.__name__,
            description=description or SynthesisXpert.__doc__,
            strategy=strategy,
            agent=EphemeralNLPAgent(
                name='SynthesisAgent',
                llm=llm,
                prompt_template=(
                    "## Task Background\n"
                    "You are given an expert response to a previous unknown query. "
                    "This response will be added to a scratchpad which tracks the query-response feedback loop. "
                    "In order to reduce length in the scratchpad, we need to compress this expert response.\n\n"
                    "## Instructions\n"
                    "Your task is to extract the essence of the expert's response by synthesing it into a short, single sentence. This will be used to index the full expert response.\n\n"
                    "## Expert Response\n"
                    "**The expert response was*:* {input}\n\n"
                    "### Output format\n"
                    "Return **only** the sentence that will be used for indexing, nothing more!!"
                ),
            ),
        )
