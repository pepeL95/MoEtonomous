from langchain_core.runnables import RunnableLambda

from agents.toolbox.toolbox import Toolbox
from agents.parsers.output import ArxivParser
from dev_tools.enums.prompts import AgentPrompts
from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
from agents.prebuilt.ephemeral_tool_agent import EphemeralToolAgent

from dev_tools.enums.llms import LLMs

from moe.base.expert import BaseExpert

class Router(BaseExpert):
    '''Router for Arxiv MoE. Decides where to go next.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')
        
        super().__init__(
            description=description or Router.__doc__,
            name=name or Router.__name__,
            strategy=strategy,
            agent=agent or RunnableLambda(lambda state: (
                f"\nAction: {QbuilderXpert.__name__}"
                f"\nAction Input: {state['input']}"
            )),
        )


class QbuilderXpert(BaseExpert):
    '''Dexterous at taking a search query and converting it into a valid JSON format for a downstream search task: searching the Arxiv api for scholar papers.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            description=description or QbuilderXpert.__doc__,
            name=name or QbuilderXpert.__name__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
                name='ArxivQbuilderAgent',
                llm=LLMs.Gemini(),
                prompt_template=AgentPrompts.Arxiv.ApiQueryBuildFewShot.value,
                output_parser=ArxivParser.ApiSearchItems.to_json(),
                system_prompt=(
                    'You are dexterous at taking in a search query and converting it '
                    'into a valid format for searching the Arxiv api for scholar papers. '
                    'Consider the user query and follow the instructions thoroughly'
                )
            ))


class SearchXpert(BaseExpert):
    '''An Arxiv api search expert. It excels at the following task: given a valid JSON query, it executes the query, searching and fetching papers from the Arxiv system.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or SearchXpert.__name__,
            description=description or SearchXpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralToolAgent(
                name='ArxivSearchAgent',
                llm=LLMs.Gemini(),
                tools=[Toolbox.Arxiv.build_query, Toolbox.Arxiv.execute_query],
                system_prompt=(
                    'You are a search expert, specialized in searching the Arxiv api for scholar papers.\n'
                    'Your task is to build a query and then execute it.\n'
                    'You have some tools at your disposal, use them wisely, in the right order.'
                ),
            )
        )


class SigmaXpert(BaseExpert):
    '''An NLP Guru specialized in summarization tasks. Useful expert when we need to synthesize information and provide insights from obtained results.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or SigmaXpert.__name__,
            description=description or SigmaXpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
                name='ArxivSigmaAgent',
                llm=LLMs.Gemini(),
                system_prompt='You are an nlp expert, specialized in summarization.',
                prompt_template=AgentPrompts.Arxiv.AbstractSigma.value
            )
        )
