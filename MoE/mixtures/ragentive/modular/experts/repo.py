import os
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

from JB007.config.debug import Debug
from JB007.toolbox.toolbox import Toolbox
from JB007.parsers.output import ArxivParser
from JB007.prompters.prompters import Prompters
from JB007.base.ephemeral_nlp_agent import EphemeralNLPAgent
from JB007.base.ephemeral_tool_agent import EphemeralToolAgent

from MoE.base.expert.base_expert import Expert
from MoE.base.mixture.base_mixture import MoE, MoEBuilder
from dev_tools.enums.embeddings import Embeddings


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
    ''''''

    def __init__(self, name, router, experts, description, strategy, verbose=Debug.Verbosity.quiet, llm=None, propmpt_parser=None):
        super().__init__(
            name=name or Pretrieval.__name__,
            router=router or None,
            experts=experts or [],
            description=description or Pretrieval.__doc__,
            strategy=strategy or None,
            verbose=verbose
        )


class Retrieval(Expert):
    '''Expert at retrieving semantically relevant documents with respect to a given query'''

    def __init__(self, agent, description, name, strategy, llm=None, propmpt_parser=None):
        super().__init__(
            name=name or Retrieval.__name__,
            description=description or Retrieval.__doc__,
            strategy=strategy,
            agent=agent or Chroma(
                collection_name='toy-embeddings',
                persist_directory=os.getenv('VDB_PATH'),
                embedding_function=Embeddings.sentence_transformers_mpnet(),
            ).as_retriever(search_kwargs={'k': 10}),
        )


class Postrieval(MoE):
    ''''''

    def __init__(self, name, router, experts, description, strategy, verbose=Debug.Verbosity.quiet, llm=None, propmpt_parser=None):
        super().__init__(
            name=name or Postrieval.__name__,
            router=router or None,
            experts=experts or [],
            description=description or Postrieval.__doc__,
            strategy=strategy or None,
            verbose=verbose
        )
