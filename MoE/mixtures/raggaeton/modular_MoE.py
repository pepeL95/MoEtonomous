from MoE.base.mixture import MoE
from MoE.base.expert import Expert
from MoE.base.router import Router
from MoE.config.debug import Debug
from MoE.xperts.expert_factory import ExpertFactory
from MoE.mixtures.raggaeton.pretrieval_MoE import PretrievalMoE
from MoE.mixtures.raggaeton.postrieval_MoE import PostrievalMoE

from typing import List

class ModularRAGMoE(MoE):
    '''Orchestrates a modular, non-naive RAG pipeline mixture of experts'''

    def __init__(self, name: str, router: Router, experts: List[Expert], description: str = None, verbose: Debug.Verbosity = Debug.Verbosity.quiet) -> None:
        super().__init__(name, router, experts, description, verbose)
        
    #########################################################################################################################

    def define_xpert_impl(self, state:MoE.State, xpert:MoE) -> dict:
        if xpert.name == PretrievalMoE.__name__:
            return self.run_preretrieval(state, xpert)
        if xpert.name == ExpertFactory.Directory.RetrieverExpert:
            return self.run_retriever_expert(state, xpert)
        if xpert.name == PostrievalMoE.__name__:
            return self.run_postretrieval(state, xpert)
        
    def run_preretrieval(self, state:MoE.State, xpert:MoE):
        output = xpert.invoke({
            'input': state['input'],
            'kwargs': {
                'topic': state['kwargs']['topic'],
            }
        })    
        state['expert_output'] = "Successfully finished the pre-retrieval step of the RAG pipeline."
        state['kwargs']['hyde'] = output['kwargs']['hyde']
        state['kwargs']['search_queries'] = output['kwargs']['search_queries']
        state['next'] = ExpertFactory.Directory.RetrieverExpert
        return state
    
    def run_retriever_expert(self, state:MoE.State, xpert:Expert):
        # n-d array with the top 10 documents per search query
        outputs = [] 
        for _q in state['kwargs']['hyde']:
            local_outputs = xpert.invoke(_q)
            outputs.append(local_outputs)

        state['expert_output'] = 'Successfully retrieved relevant documents.'
        state['kwargs']['context'] = outputs
        state['next'] = PostrievalMoE.__name__
        return state
    
    def run_postretrieval(self, state:MoE.State, xpert:MoE):
        output = xpert.invoke({
            'input': state['input'],
            'kwargs': state['kwargs']
        })   
        state['expert_output'] = output['expert_output']
        state['next'] = 'END'
        return state