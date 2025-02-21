from langchain_core.output_parsers import JsonOutputParser

from dev_tools.enums.llms import LLMs
from dev_tools.enums.prompts import AgentPrompts

from agents.tools.toolbox import ArxivToolbox
from agents.tools.toolschemas import ArxivSchema
from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
from agents.prebuilt.ephemeral_tool_agent import EphemeralToolAgent

from moe.annotations.core import Expert
from moe.prebuilt.arxiv.strategies import QueryStrategy, SearchStrategy, SigmaStrategy


@Expert(QueryStrategy)
class QbuilderXpert:
    '''Dexterous at taking a search query and converting it into a valid JSON format for a downstream search task: searching the Arxiv api for scholar papers.'''
    agent = EphemeralNLPAgent(
        name='ArxivQbuilderAgent',
        llm=LLMs.Gemini(),
        prompt_template=AgentPrompts.Arxiv.ApiQueryBuildFewShot.value,
        output_parser=JsonOutputParser(pydantic_object=ArxivSchema.ApiSearchItems),
        system_prompt=(
            'You are dexterous at taking in a search query and converting it '
            'into a valid format for searching the Arxiv api for scholar papers. '
            'Consider the user query and follow the instructions thoroughly'
        )
    )

@Expert(SearchStrategy)
class SearchXpert:
    '''An Arxiv api search expert. It excels at the following task: given a valid JSON query, it executes the query, searching and fetching papers from the Arxiv system.'''
    agent = EphemeralToolAgent(
        name='ArxivSearchAgent',
        llm=LLMs.Gemini(),
        tools=[ArxivToolbox.build_query_tool, ArxivToolbox.execute_query_tool],
        system_prompt=(
            'You are a search expert, specialized in searching the Arxiv api for scholar papers.\n'
            'Your task is to build a query and then execute it.\n'
            'You have some tools at your disposal, use them wisely, in the right order.'
        ),
    )
        


@Expert(SigmaStrategy)
class SigmaXpert:
    '''An NLP Guru specialized in summarization tasks. Useful expert when we need to synthesize information and provide insights from obtained results.'''
    agent = EphemeralNLPAgent(
        name='ArxivSigmaAgent',
        llm=LLMs.Gemini(),
        system_prompt='You are an nlp expert, specialized in summarization.',
        prompt_template=AgentPrompts.Arxiv.AbstractSigma.value
    )


################################################################################################################


class Factory:
    class Dir:
        QbuilderXpert: str = 'QbuilderXpert'
        SearchXpert: str = 'SearchXpert'
        SigmaXpert: str = 'SigmaXpert'

    @staticmethod
    def get(expert_name: str, agent=None):
        # Factory pattern
        if expert_name == Factory.Dir.QbuilderXpert:
            return QbuilderXpert(agent=agent, strategy=QueryStrategy())
        if expert_name == Factory.Dir.SearchXpert:
            return SearchXpert(agent=agent, strategy=SearchStrategy())
        if expert_name == Factory.Dir.SigmaXpert:
            return SigmaXpert(agent=agent, strategy=SigmaStrategy())

        raise ValueError(f'No expert by name `{expert_name}` exists.')