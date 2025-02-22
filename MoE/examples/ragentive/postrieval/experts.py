from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent

from moe.annotations.core import Expert
from moe.examples.ragentive.postrieval.strategies import ContextStrategy, RerankingStrategy

from RAG.postrieval.cross_encoding_reranker import CrossEncodingReranker

from dev_tools.enums.llms import LLMs
from dev_tools.enums.cross_encodings import CrossEncodings


@Expert(RerankingStrategy)
class RerankingExpert:
    '''A master at ranking the relevance of retrieved documents with respect to a given query. It usually does its work after the RetrieverExpert'''
    agent = CrossEncodingReranker(
            cross_encoder=CrossEncodings.LocalMiniLM()
        ).as_reranker(rerank_kwargs={'k': 4})


@Expert(ContextStrategy)
class ContextExpert:
    '''Master at giving informed answers. It uses a given context to augment its knowledge.'''
    agent = EphemeralNLPAgent(
        name='ContextAgent',
        llm=LLMs.Gemini(),
        system_prompt="You are an expert at giving informed answers.",
        prompt_template=(
            "Use the following topic and pieces of retrieved context to enhance your knowledge\n"
            "Answer the user query as best as possible\n"
            "If you don't know the answer, try your best to answer anyways. \n"
            "Be comprehensive with your answers.\n\n"
            "### Topic\n"
            "{topic}. \n\n"
            "### Query:\n"
            "{input}\n\n"
            "### Context:\n"
            "{context}\n\n"
            "### Answer:\n"
        ),
    )

####################################################################################################

class Factory:
    class Dir:
        RerankingExpert: str =  'RerankingExpert'
        ContextExpert: str = 'ContextExpert'

    @staticmethod
    def get(expert_name:str, **kwargs):
        if expert_name == Factory.Dir.RerankingExpert:
            return RerankingExpert(**kwargs)
        if expert_name == Factory.Dir.ContextExpert:
            return ContextExpert(**kwargs)

        raise ValueError(f'No expert by name {expert_name} exists.')

