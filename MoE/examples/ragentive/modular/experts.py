import os
from langchain_chroma import Chroma

from dev_tools.enums.embeddings import Embeddings

from moe.base.mixture import BaseMoE
from moe.annotations.core import Expert, Deterministic, MoE
from moe.examples.ragentive.pretrieval.experts import Factory as PreFactory 
from moe.examples.ragentive.postrieval.experts import Factory as PostFactory 
from moe.examples.ragentive.modular.strategies import PostrievalStrategy, PretrievalStrategy, RetrievalStrategy


@MoE(PretrievalStrategy)
@Deterministic('QueryAugmentationExpert')
class Pretrieval:
    '''A master at orchestrating the pre-retrieval step of a Retrieval Augmented Generation (RAG) pipeline. It returns a hypothetical answer that must be given to the PostrievalMoE.Use this expert at the beginning of the pipeline.'''
    experts = [
        PreFactory.get(PreFactory.Dir.QueryAugmentationExpert),
        PreFactory.get(PreFactory.Dir.HydeExpert)
    ]


@Expert(RetrievalStrategy)
class Retrieval:
    '''Expert at retrieving semantically relevant documents with respect to a given query'''
    agent = Chroma(
        collection_name='toy-embeddings',
        persist_directory=os.getenv('VDB_PATH'),
        embedding_function=Embeddings.LocalMPNetBaseV2(),
    ).as_retriever(search_kwargs={'k': 10})


@MoE(PostrievalStrategy)
@Deterministic('RerankingExpert')
class Postrieval(BaseMoE):
    '''Expert at coordinating the post-retrieval step of a Retrieval Augmented Generation (RAG) pipeline. Use this expert when the pre-retrival step is done. You may END after this expert has responded.'''
    experts = [
        PostFactory.get(expert_name=PostFactory.Dir.RerankingExpert),
        PostFactory.get(expert_name=PostFactory.Dir.ContextExpert),    
    ]
        


####################################################################################################

class Factory:
    class Dir:
        Pretrieval: str =  'Pretrieval'
        Retrieval: str = 'Retrieval'
        Postrieval: str =  'Postrieval'

    @staticmethod
    def get(expert_name:str, **kwargs):
        if expert_name == Factory.Dir.Pretrieval:
            return Pretrieval(**kwargs)
        if expert_name == Factory.Dir.Retrieval:
            return Retrieval(**kwargs)
        if expert_name == Factory.Dir.Postrieval:
            return Postrieval(**kwargs)

        raise ValueError(f'No expert by name {expert_name} exists.')
