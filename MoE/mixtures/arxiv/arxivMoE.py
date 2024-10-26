
from typing import List

from MoE.base.expert import Expert
from MoE.base.mixture import MoE
from MoE.base.router import Router
from MoE.config.debug import Debug

from JB007.parsers.generic import StringParser

class ArxivMoE(MoE):
    def __init__(self, name: str, router: Router, experts: List[Expert], description: str = None, verbose: Debug.Verbosity = Debug.Verbosity.quiet) -> None:
        super().__init__(name, router, experts, description, verbose)

    def define_xpert_impl(self, state: MoE.State, xpert: Expert) -> dict:
        if xpert.name == 'ArxivQbuilderXpert':
            return self.run_arxiv_qbuilder(state, xpert)
        
        if xpert.name == 'ArxivSearchXpert':
            return self.run_arxiv_searchxpert(state, xpert)
        
        if xpert.name == 'ArxivSigmaXpert':
            return self.run_arxiv_sigmaxpert(state, xpert)
        
    
    def run_arxiv_qbuilder(self, state:MoE.State, xpert:Expert):
        output = xpert.invoke({'input': state['expert_input']})

        state['expert_output'] = 'Successfully built json query: ' + str(output)
        state['kwargs']['apiJson'] = output
        state['next'] = 'ArxivSearchXpert'
        return state
    
    def run_arxiv_searchxpert(self, state:MoE.State, xpert:Expert):
        output = xpert.invoke({'input': state['kwargs']['apiJson']})

        state['expert_output'] = 'Successfully fetched papers from Arxiv.'
        state['kwargs']['papers'] = output
        state['next'] = 'ArxivSigmaXpert'
        return state
    
    def run_arxiv_sigmaxpert(self, state:MoE.State, xpert:Expert):
        outputs = []
        for paper in state['kwargs']['papers']:
            output = xpert.invoke({'input': paper})
            outputs.append(output)

        state['expert_output'] = StringParser.from_array(outputs)
        state['next'] = 'END'
        return state