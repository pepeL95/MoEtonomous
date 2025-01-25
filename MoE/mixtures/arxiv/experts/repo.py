from langchain_core.runnables import RunnableLambda

from JB007.toolbox.toolbox import Toolbox
from JB007.parsers.output import ArxivParser
from JB007.prompters.prompters import Prompters
from JB007.base.ephemeral_nlp_agent import EphemeralNLPAgent
from JB007.base.ephemeral_tool_agent import EphemeralToolAgent

from MoE.base.expert.base_expert import Expert


class Router(Expert):
    '''Router for Arxiv MoE. Decides where to go next.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None, prompt_parser=None):
        super().__init__(
            description=description or Router.__doc__,
            name=name or Router.__name__,
            strategy=strategy,
            agent=agent or RunnableLambda(lambda state: (
                f"\nAction: {QbuilderXpert.__name__}"
                f"\nAction Input: {state['input']}"
            )),
        )


class QbuilderXpert(Expert):
    '''Dexterous at taking a search query and converting it into a valid JSON format for a downstream search task: searching the Arxiv api for scholar papers.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None, prompt_parser=None):
        if llm is None:
            raise ValueError('LLM cannot be None')

        super().__init__(
            description=description or QbuilderXpert.__doc__,
            name=name or QbuilderXpert.__name__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
                llm=llm,
                name='ArxivQbuilderAgent',
                prompt_parser=prompt_parser,
                prompt_template=Prompters.Arxiv.ApiQueryBuildFewShot(),
                output_parser=ArxivParser.ApiSearchItems.to_json(),
                system_prompt=(
                    'You are an dexterous at taking a search query and converting it '
                    'into a valid format for searching the Arxiv api for scholar papers. '
                    'Consider the user query and follow the instructions thoroughly'
                )
            ))


class SearchXpert(Expert):
    '''An Arxiv api search expert. It excels at the following task: given a valid JSON query, it executes the query, searching and fetching papers from the Arxiv system.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None, prompt_parser=None):
        if llm is None:
            raise ValueError('LLM cannot be None')

        super().__init__(
            name=name or SearchXpert.__name__,
            description=description or SearchXpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralToolAgent(
                llm=llm,
                name='ArxivSearchAgent',
                prompt_parser=prompt_parser,
                tools=[Toolbox.Arxiv.build_query, Toolbox.Arxiv.execute_query],
                system_prompt=(
                    'You are a search expert, specialized in searching the Arxiv api for scholar papers.\n'
                    'Your task is to build a query and then execute it.\n'
                    'You have some tools at your disposal, use them wisely, in the right order.'
                ),
            )
        )


class SigmaXpert(Expert):
    '''An NLP Guru specialized in summarization tasks. Useful expert when we need to synthesize information and provide insights from obtained results.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None, prompt_parser=None):
        if llm is None:
            raise ValueError('LLM cannot be None')

        super().__init__(
            name=name or SigmaXpert.__name__,
            description=description or SigmaXpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
                llm=llm,
                prompt_parser=prompt_parser,
                name='ArxivSigmaAgent',
                system_prompt='You are an nlp expert, specialized in summarization.',
                prompt_template=Prompters.Arxiv.AbstractSigma()
            )
        )
