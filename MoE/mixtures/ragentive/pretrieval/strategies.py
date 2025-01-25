from MoE.xperts.expert_factory import ExpertFactory
from MoE.base.expert import Expert
from MoE.base.expert.lazy_expert import LazyExpert
from MoE.config.debug import Debug
from MoE.base.mixture.base_mixture import MoE

from typing import List


class PretrievalMoE(MoE):
    '''Modular class for handling the  pre-retrieval step of a non-naive RAG pipeline'''

    def __init__(self, name: str, router: LazyExpert, experts: List[Expert], description: str = None, verbose: Debug.Verbosity = Debug.Verbosity.quiet) -> None:
        super().__init__(name, router, experts, description, verbose)

    #########################################################################################################################

    def execute_strategy(self, state: MoE.State, xpert: Expert) -> dict:
        if xpert.name == ExpertFactory.Directory.QueryXtractionXpert:
            return self.run_queryXtractionXpert(state, xpert)
        if xpert.name == ExpertFactory.Directory.HyDExpert:
            return self.run_HyDEXpert(state, xpert)

    def run_queryXtractionXpert(self, state: MoE.State, xpert: Expert) -> dict:
        output = xpert.invoke({
            'input': state['expert_input'],
            'topic': state['kwargs'].get('topic', 'No specific topic, go ahead and imply it as best as you can from the query'),
        })
        state['expert_output'] = "Successfully enhanced the queries."
        state['kwargs']['search_queries'] = output['search_queries']
        state['next'] = ExpertFactory.Directory.HyDExpert
        return state

    def run_HyDEXpert(self, state: MoE.State, xpert: Expert) -> dict:
        outputs = []
        for _q in state['kwargs']['search_queries']:
            local_output = xpert.invoke({
                'input': _q,
                'topic': state['kwargs'].get('topic', 'No specific topic, go ahead and imply it as best as you can from the query'),
                'context':  state['kwargs'].get('context', ''),
            })
            outputs.append(local_output)
        state['expert_output'] = "Successfully generated hypothetical documents."
        state['kwargs']['hyde'] = outputs
        state['next'] = 'END'
        return state
