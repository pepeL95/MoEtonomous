from MoE.xperts.expert_factory import ExpertFactory
from MoE.base.expert import Expert
from MoE.base.router import Router
from MoE.config.debug import Debug
from MoE.base.mixture import MoE

from JB007.parsers.output import StringParser

from typing import List

class PostrievalMoE(MoE):
    '''Modular class for handling the  post-retrieval step of a non-naive RAG pipeline'''

    def __init__(self, name: str, router: Router, experts: List[Expert], description: str = None, verbose: Debug.Verbosity = Debug.Verbosity.quiet) -> None:
        super().__init__(name, router, experts, description, verbose)
    
    #########################################################################################################################

    def define_xpert_impl(self, state: MoE.State, xpert: Expert) -> dict:
        if xpert.name == ExpertFactory.Directory.RerankingExpert:
            return self.run_reranking_expert(state, xpert)
        if xpert.name == ExpertFactory.Directory.ContextExpert:
            return self.run_context_expert(state, xpert)
    
    ########################################### POST-RETRIEVAL ##############################################################################

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
        
        state['kwargs']['context'] = StringParser.from_langdocs(list(outputs))
        state['expert_output'] = state['kwargs']['context']
        state['next'] = ExpertFactory.Directory.ContextExpert
        return state
    
    def run_context_expert(self, state:MoE.State, xpert:Expert):
        output = xpert.invoke({
            'input': state['expert_input'],
            'topic': state['kwargs'].get('topic', 'No specific topic, go ahead and imply it as best as you can from the query'),
            'context': state['kwargs']['context']
        })

        state['expert_output'] = output
        state['next'] = 'END'
        return state