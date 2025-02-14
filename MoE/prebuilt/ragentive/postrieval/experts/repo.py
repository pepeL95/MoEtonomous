from moe.base.expert import BaseExpert

from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent

from RAG.postrieval.cross_encoding_reranker import CrossEncodingReranker

from dev_tools.enums.llms import LLMs
from dev_tools.enums.cross_encodings import CrossEncodings

from langchain_core.runnables import RunnableLambda


class Router(BaseExpert):
    def __init__(self, agent=None, description=None, name=None, strategy=None):
        super().__init__(
            description=description or Router.__doc__,
            name=name or Router.__name__,
            strategy=strategy,
            agent=agent or RunnableLambda(lambda state: (
                f"\nAction: {RerankingExpert.__name__}"
                f"\nAction Input: {state['input']}"
            )),
        )


class RerankingExpert(BaseExpert):
    '''A master at ranking the relevance of retrieved documents with respect to a given query. It usually does its work after the RetrieverExpert'''

    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or RerankingExpert.__name__,
            description=description or RerankingExpert.__doc__,
            strategy=strategy,
            agent=agent or CrossEncodingReranker(
                cross_encoder=CrossEncodings.LocalMiniLM()
            ).as_reranker(rerank_kwargs={'k': 4})
        )


class ContextExpert(BaseExpert):
    '''Master at giving informed answers. It uses a given context to augment its knowledge.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None):        
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or ContextExpert.__name__,
            description=description or ContextExpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
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
        )
