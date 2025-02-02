from agents.base.ephemeral_nlp_agent import EphemeralNLPAgent

from MoE.base.expert.base_expert import BaseExpert
from MoE.prompts.prompt_repo import PromptRepo
from MoE.base.expert.lazy_expert import LazyExpert

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
                f"\nAction: {PlanningXpert.__name__}"
                f"\nAction Input: {state['input']}"
            )),
        )


class IntentXtractor(BaseExpert):
    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or IntentXtractor.__name__,
            description=description or IntentXtractor.__doc__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
                name='IntentXtractionAgent',
                llm=LLMs.Phi35(),
                prompt_parser=PromptParsers.Identity(),
                # prompt_parser=PromptParsers.Phi35(),
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


class PlanningXpert(LazyExpert):
    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or PlanningXpert.__name__,
            description=description or PlanningXpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
                name='PlanningAgent',
                llm=LLMs.Gemini(),
                prompt_parser=PromptParsers.Identity(),
                prompt_template=PromptRepo.MoE_ReAct(),
                system_prompt=(
                    '## System Information\n'
                    'You are a strategic decision-making agent, who excels at immediate action planning. '
                    'Keep your plan concise.\n'
                    # 'Consider the following chat history (if any).'
                ),
            ),
        )


class SynthesisXpert(BaseExpert):
    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or SynthesisXpert.__name__,
            description=description or SynthesisXpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
                name='SynthesisAgent',
                llm=LLMs.Phi35(),
                prompt_parser=PromptParsers.Identity(),
                # prompt_parser=PromptParsers.Phi35(),
                prompt_template=(
                    "## Instructions\n"
                    "Generate a one-sentence synthesis of the following. Make sure to include key terms in your synthesis.\n\n"
                    "## Input\n"
                    "{input}"
                ),
            ),
        )
