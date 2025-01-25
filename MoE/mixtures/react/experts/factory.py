from JB007.base.ephemeral_nlp_agent import EphemeralNLPAgent

from MoE.base.expert.base_expert import Expert
from MoE.prompts.prompt_repo import PromptRepo
from MoE.base.expert.lazy_expert import LazyExpert
from MoE.mixtures.react.strategies import PlanningStrategy, SynthesisStrategy, IntentXtractStrategy, RouterStrategy

from langchain_core.runnables import RunnableLambda


class Router:
    @staticmethod
    def get():
        return Expert(
            name='Router',
            description='React agent that autonimously devises a plan and executes it accordingly',
            strategy=RouterStrategy(),
            agent=RunnableLambda(lambda state: (
                f"\nAction: {ReActDirectory.PlanningXpert}"
                f"\nAction Input: {state['input']}"
            )),
        )


class IntentXtractor:
    @staticmethod
    def get(llm, prompt_parser):
        return Expert(
            name='IntentXtractor',
            description='Expert at compiling a query into a thorough insight which synthesizes what it is to be achieved.',
            strategy=IntentXtractStrategy(),
            agent=EphemeralNLPAgent(
                name='IntentXtractionAgent',
                llm=llm,
                prompt_parser=prompt_parser,
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
            )
        )


class PlanningXpert:
    @staticmethod
    def get(llm, prompt_parser):
        return LazyExpert(
            name='PlanningXpert',
            description='Master at generating a plan of what to do next, given a premise.',
            strategy=PlanningStrategy(),
            stop_tokens=['\nExpert Response:'],
            agent=EphemeralNLPAgent(
                name='PlanningAgent',
                llm=llm,
                prompt_parser=prompt_parser,
                prompt_template=PromptRepo.MoE_ReAct(),
                system_prompt=(
                    '## System Information\n'
                    'You are a strategic decision-making agent, who excels at immediate action planning. '
                    'Keep your plan concise.\n'
                    # 'Consider the following chat history (if any).'
                ),
            )
        )


class SynthesisXpert:
    @staticmethod
    def get(llm, prompt_parser):
        return Expert(
            name='SynthesisXpert',
            description=None,
            strategy=SynthesisStrategy(),
            agent=EphemeralNLPAgent(
                name='SynthesisAgent',
                llm=llm,
                prompt_parser=prompt_parser,
                prompt_template=(
                    "## Instructions\n"
                    "Generate a one-sentence synthesis of the following. Make sure to include key terms in your synthesis.\n\n"
                    "## Input\n"
                    "{input}"
                ),
            )
        )


class ReActDirectory:
    Router: str = Router.__name__
    IntentXtractor: str = IntentXtractor.__name__
    PlanningXpert: str = PlanningXpert.__name__
    SynthesisXpert: str = SynthesisXpert.__name__
    ActionExecXpert: str = 'ActionExecXpert'


class ReActFactory:
    @staticmethod
    def get(expert_name: str, llm, prompt_parser=None):
        if expert_name == ReActDirectory.Router:
            return Router.get()
        if expert_name == ReActDirectory.IntentXtractor:
            return IntentXtractor.get(llm=llm, prompt_parser=prompt_parser)
        if expert_name == ReActDirectory.PlanningXpert:
            return PlanningXpert.get(llm=llm, prompt_parser=prompt_parser)
        if expert_name == ReActDirectory.SynthesisXpert:
            return SynthesisXpert.get(llm=llm, prompt_parser=prompt_parser)
        if expert_name == ReActDirectory.ActionExecXpert:
            return None
        raise ValueError(f'No expert by name {expert_name} exists.')
