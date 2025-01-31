from JB007.parsers.output import StringParser
from MoE.base.mixture.base_mixture import MoE
from MoE.base.strategy.expert.base_strategy import BaseExpertStrategy
from MoE.mixtures.arxiv.experts.factory import ArxivDirectory


class RouterStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke(state)
        state = {'expert_output': output}
        return state


class QueryStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({'input': state['expert_input']})

        state['expert_output'] = 'Successfully built json query: ' + str(output)
        state['kwargs']['apiJson'] = output
        state['next'] = ArxivDirectory.SearchXpert
        return state


class SearchStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({'input': state['kwargs']['apiJson']})

        state['expert_output'] = 'Successfully fetched papers from Arxiv.'
        state['kwargs']['papers'] = output
        state['next'] = ArxivDirectory.SigmaXpert
        return state


class SigmaStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        outputs = []
        for paper in state['kwargs']['papers']:
            output = expert.invoke({'input': paper})
            outputs.append(output)

        state['expert_output'] = StringParser.from_array(outputs)
        state['next'] = MoE.FINISH
        return state
