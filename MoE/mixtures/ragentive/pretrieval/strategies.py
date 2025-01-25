from MoE.base.mixture.base_mixture import MoE
from MoE.base.strategy.expert.base_strategy import BaseExpertStrategy
from MoE.mixtures.ragentive.pretrieval.experts.factory import PretrievalDirectory


class RouterStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke(state)
        return {'expert_output': output}


class QueryAugmentationStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['expert_input'],
            'topic': state['kwargs'].get('topic', 'No specific topic, go ahead and imply it as best as you can from the query'),
        })
        state['expert_output'] = "Successfully enhanced the queries."
        state['kwargs']['search_queries'] = output['search_queries']
        state['next'] = PretrievalDirectory.HydeExpert
        return state


class HydeStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        outputs = []
        for _q in state['kwargs']['search_queries']:
            local_output = expert.invoke({
                'input': _q,
                'topic': state['kwargs'].get('topic', 'No specific topic, go ahead and imply it as best as you can from the query'),
                'context':  state['kwargs'].get('context', ''),
            })
            outputs.append(local_output)
        state['expert_output'] = "Successfully generated hypothetical documents."
        state['kwargs']['hyde'] = outputs
        state['next'] = MoE.FINISH
        return state
