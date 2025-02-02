from agents.parsers.output import StringParser
from moe.base.mixture import BaseMoE
from moe.base.strategies import BaseExpertStrategy
from moe.prebuilt.arxiv.experts.factory import ArxivDirectory


class RouterStrategy(BaseExpertStrategy):
    def execute(self, expert, state:BaseMoE.State):
        output = expert.invoke(state)
        state = {'expert_output': output}
        return state


class QueryStrategy(BaseExpertStrategy):
    def execute(self, expert, state:BaseMoE.State):
        output = expert.invoke({'input': state['expert_input']})

        state['expert_output'] = 'Successfully built json query: ' + str(output)
        state['kwargs']['apiJson'] = output
        state['next'] = ArxivDirectory.SearchXpert
        return state


class SearchStrategy(BaseExpertStrategy):
    def execute(self, expert, state:BaseMoE.State):
        output = expert.invoke({'input': state['kwargs']['apiJson']})

        state['expert_output'] = 'Successfully fetched papers from Arxiv.'
        state['kwargs']['papers'] = output
        state['next'] = ArxivDirectory.SigmaXpert
        return state


class SigmaStrategy(BaseExpertStrategy):
    def execute(self, expert, state:BaseMoE.State):
        outputs = []
        for paper in state['kwargs']['papers']:
            output = expert.invoke({'input': paper})
            outputs.append(output)

        state['expert_output'] = StringParser.from_array(outputs)
        state['next'] = BaseMoE.FINISH
        return state
