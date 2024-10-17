from MoE.xperts.expert_factory import ExpertFactory
from MoE.base.expert import Expert
from MoE.base.router import Router
from MoE.config.debug import Debug
from MoE.base.mixture import MoE

from typing import List

class PostrievalMoE:
    '''Modular class for handling the  post-retrieval step of a non-naive RAG pipeline'''

    def __init__(self, name: str, router: Router, experts: List[Expert], description: str = None, verbose: Debug.Verbosity = Debug.Verbosity.quiet) -> None:
        super().__init__(name, router, experts, description, verbose)
    
    #########################################################################################################################

    def define_xpert_impl(self, state: MoE.State, xpert: Expert) -> dict:
        if xpert.name == ExpertFactory.Directory.RetrieverExpert:
            return self.run_retriever_expert(state, xpert)
        if xpert.name == ExpertFactory.Directory.RerankingExpert:
            return self.run_reranking_expert(state, xpert)
        if xpert.name == ExpertFactory.Directory.ContextExpert:
            return self.run_content_expert(state, xpert)
    
    ########################################### POST-RETRIEVAL ##############################################################################

    def run_retriever_expert(self, state:MoE.State, xpert:Expert):
        # # n-d array with the top 10 documents per search query
        outputs = [] 
        for _q in state['kwargs']['hyde']:
            local_outputs = xpert.invoke(_q)
            outputs.append(local_outputs)

        state['expert_output'] = 'Successfully retrieved relevant documents.'
        state['kwargs']['context'] = outputs
        state['next'] = ExpertFactory.Directory.RerankingExpert
        return state
    

    def run_reranking_expert(self, state:MoE.State, xpert:Expert):
        if not 'context' in state['kwargs']:
            raise ValueError('Context documents not provided to the reranker')

        # Top k documents per query, where k is defined at expert init time
        outputs = set()
        for _q, docs in zip(state['kwargs']['search_queries'], state['kwargs']['context']):
            local_outputs = xpert.invoke({
                'input': _q,
                'context': docs,
            })
            outputs.update(local_outputs)
        
        # state['kwargs']['context'] = format_documents(list(outputs))
        state['kwargs']['context'] = outputs
        state['expert_output'] = state['kwargs']['context']
        state['next'] = ExpertFactory.Directory.ContextExpert
        return state