import os
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

from JB007.config.debug import Debug

from MoE.base.expert.base_expert import Expert
from MoE.base.mixture.base_mixture import MoE
from MoE.mixtures.ragentive.postrieval.experts.factory import PostrievalDirectory, PostrievalFactory
from MoE.mixtures.ragentive.pretrieval.experts.factory import PretrievalDirectory, PretrievalFactory


from dev_tools.enums.embeddings import Embeddings
from dev_tools.enums.llms import LLMs
from dev_tools.enums.prompt_parsers import PromptParsers


class Router(Expert):
    '''Router for a modular, agentive RAG MoE. Decides where to go next.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None, prompt_parser=None):
        super().__init__(
            description=description or Router.__doc__,
            name=name or Router.__name__,
            strategy=strategy,
            agent=agent or RunnableLambda(lambda state: (
                f"\nAction: {Pretrieval.__name__}"
                f"\nAction Input: {state['input']}"
            )),
        )


class Pretrieval(MoE):
    '''A master at orchestrating the pre-retrieval step of a Retrieval Augmented Generation (RAG) pipeline. It returns a hypothetical answer that must be given to the PostrievalMoE.Use this expert at the beginning of the pipeline.'''

    def __init__(self, name=None, router=None, experts=None, description=None, strategy=None, verbose=Debug.Verbosity.quiet, llm=None, prompt_parser=None):
        if strategy is None:
            raise ValueError("strategy cannot be None")

        super().__init__(
            name=name or Pretrieval.__name__,
            router=router or PretrievalFactory.get(expert_name=PretrievalDirectory.Router, llm=None, prompt_parser=None),
            experts=experts or [
                PretrievalFactory.get(PretrievalDirectory.QueryAugmentationExpert, llm=LLMs.Gemini(), prompt_parser=PromptParsers.Identity()),
                PretrievalFactory.get(PretrievalDirectory.HydeExpert, llm=LLMs.Gemini(), prompt_parser=PromptParsers.Identity())
            ],
            description=description or Pretrieval.__doc__,
            strategy=strategy,
            verbose=verbose
        )


class Retrieval(Expert):
    '''Expert at retrieving semantically relevant documents with respect to a given query'''
        
    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None, prompt_parser=None):
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


class Postrieval(MoE):
    '''Expert at coordinating the post-retrieval step of a Retrieval Augmented Generation (RAG) pipeline. Use this expert when the pre-retrival step is done. You may END after this expert has responded.'''

    def __init__(self, name=None, router=None, experts=None, description=None, strategy=None, verbose=Debug.Verbosity.quiet, llm=None, prompt_parser=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')
        
        super().__init__(
            name=name or Postrieval.__name__,
            router=router or PostrievalFactory.get(expert_name=PostrievalDirectory.Router, llm=None, prompt_parser=None),
            experts=experts or [
                PostrievalFactory.get(expert_name=PostrievalDirectory.RerankingExpert, llm=None, prompt_parser=None),
                PostrievalFactory.get(expert_name=PostrievalDirectory.ContextExpert, llm=LLMs.Gemini(), prompt_parser=PromptParsers.Identity()),    
            ],
            description=description or Postrieval.__doc__,
            strategy=strategy,
            verbose=verbose,
        )
