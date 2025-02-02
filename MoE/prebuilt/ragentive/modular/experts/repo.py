import os
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

from agents.config.debug import Debug

from moe.base.expert import BaseExpert
from moe.base.mixture import BaseMoE
from moe.prebuilt.ragentive.postrieval.experts.factory import PostrievalDirectory, PostrievalFactory
from moe.prebuilt.ragentive.pretrieval.experts.factory import PretrievalDirectory, PretrievalFactory

from dev_tools.enums.embeddings import Embeddings


class Router(BaseExpert):
    '''Router for a modular, agentive RAG MoE. Decides where to go next.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError("strategy cannot be None")
        
        super().__init__(
            description=description or Router.__doc__,
            name=name or Router.__name__,
            strategy=strategy,
            agent=agent or RunnableLambda(lambda state: (
                f"\nAction: {Pretrieval.__name__}"
                f"\nAction Input: {state['input']}"
            )),
        )


class Pretrieval(BaseMoE):
    '''A master at orchestrating the pre-retrieval step of a Retrieval Augmented Generation (RAG) pipeline. It returns a hypothetical answer that must be given to the PostrievalMoE.Use this expert at the beginning of the pipeline.'''

    def __init__(self, name=None, router=None, experts=None, description=None, strategy=None, verbose=Debug.Verbosity.quiet):
        if strategy is None:
            raise ValueError("strategy cannot be None")

        super().__init__(
            name=name or Pretrieval.__name__,
            router=router or PretrievalFactory.get(expert_name=PretrievalDirectory.Router),
            experts=experts or [
                PretrievalFactory.get(PretrievalDirectory.QueryAugmentationExpert),
                PretrievalFactory.get(PretrievalDirectory.HydeExpert)
            ],
            description=description or Pretrieval.__doc__,
            strategy=strategy,
            verbose=verbose
        )


class Retrieval(BaseExpert):
    '''Expert at retrieving semantically relevant documents with respect to a given query'''
        
    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')
        
        super().__init__(
            name=name or Retrieval.__name__,
            description=description or Retrieval.__doc__,
            strategy=strategy,
            agent=agent or Chroma(
                collection_name='toy-embeddings',
                persist_directory=os.getenv('VDB_PATH'),
                embedding_function=Embeddings.sentence_transformers_mpnet(),
            ).as_retriever(search_kwargs={'k': 10})
        )


class Postrieval(BaseMoE):
    '''Expert at coordinating the post-retrieval step of a Retrieval Augmented Generation (RAG) pipeline. Use this expert when the pre-retrival step is done. You may END after this expert has responded.'''

    def __init__(self, name=None, router=None, experts=None, description=None, strategy=None, verbose=Debug.Verbosity.quiet):
        if strategy is None:
            raise ValueError('strategy cannot be None')
        
        super().__init__(
            name=name or Postrieval.__name__,
            router=router or PostrievalFactory.get(expert_name=PostrievalDirectory.Router),
            experts=experts or [
                PostrievalFactory.get(expert_name=PostrievalDirectory.RerankingExpert),
                PostrievalFactory.get(expert_name=PostrievalDirectory.ContextExpert),    
            ],
            description=description or Postrieval.__doc__,
            strategy=strategy,
            verbose=verbose,
        )
