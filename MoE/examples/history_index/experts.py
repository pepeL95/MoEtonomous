from dev_tools.enums.embeddings import Embeddings
from langchain_core.output_parsers import JsonOutputParser
from dev_tools.enums.llms import LLMs
from moe.annotations.core import Expert
from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
from moe.examples.history_index.index_service import IndexService
from moe.examples.history_index.strategies import IndexingStrategy, MetadataXStrategy, ResponseStrategy, ThoughtStrategy



@Expert(ThoughtStrategy)
class ThoughtXpert:
    '''Generates insightful thoughts to reveal intent and enrich the downstream answer.'''
    agent = EphemeralNLPAgent(
        name='ThoughtXpert',
        llm=LLMs.Gemini(),
        system_prompt=(
            '## Instructions'
            'You are a highly analytical and methodical reasoning agent. Your task is to carefully examine every user query and produce '
            'a comprehensive, step-by-step explanation of your reasoning process. Instead of providing a final answer to the query, your '
            'output should consist solely of your detailed chain-of-thought. This chain-of-thought should include:\n'
            '1.	Query Understanding: A clear breakdown of what the query is asking.\n'
            '2.	Decomposition: Identification of key components, underlying assumptions, and any ambiguities.\n'
            '3.	Step-by-Step Reasoning: A thorough, sequential explanation of how you analyze each part of the query, including any considerations or alternative interpretations.\n'
            '4.	Verification: An overview of how you verify and validate your reasoning steps.\n\n'
            'Your entire response should be just this detailed reasoning process, without any concluding final answer or summary.'
        ),
        prompt_template=(
            '## User Query\n'
            '{input}'
        ),
    )

@Expert(ResponseStrategy)
class ResponseXpert:
    '''Generates a final response based on the query and reasoning provided.'''
    agent = EphemeralNLPAgent(
        name='ResponseXpert',
        llm=LLMs.Gemini(),
        system_prompt='You are an expert analyst.',
        prompt_template='''## Instructions
            You are an expert in generating accurate, well-reasoned responses to user queries.
            Your input will contain both the user’s query and a detailed chain-of-thought reasoning process.
            Your task is to use this reasoning process to produce a clear and precise response. Your response should:
            - 1. Directly answer the query while leveraging the provided reasoning.
            - 2. Avoid unnecessary repetition or generic statements.
            - 3. Ensure factual accuracy and clarity.

            Respond only with the final answer based on the provided reasoning.

            **Input**
            {input}
            '''
    )

@Expert(MetadataXStrategy)
class MetadataXpert:
    '''Generates metadata for indexing and future retrieval.'''
    agent = EphemeralNLPAgent(
        name='MetadataXpert',
        llm=LLMs.Gemini(),
        system_prompt="You are an expert analyst",
        prompt_template='''## Instructions
            You are responsible for generating structured metadata for indexing and retrieval purposes.
            Your input will contain a user query, the corresponding reasoning, and the generated response.
            Your task is to
            - 1. Extract relevant semantic metadata from the input.
            - 2. Ensure the metadata accurately reflects the content of the response.
            - 3. Return only the structured metadata in JSON format. Use the following schema:

            {{
                "user_intent": "<user intent inferred from the query and reasoning steps>",
                "topic": "<relevant and descriptive topic, according to the response>",
                "category": "<Broader semantic category>"
            }}

            **Input**
            {input}''',
        output_parser=JsonOutputParser()
    )

@Expert(IndexingStrategy)
class IndexXpert:
    '''Indexes responses and metadata into the document retrieval system.'''
    agent = IndexService('.dev_chroma_db', Embeddings.GoogleText004())

