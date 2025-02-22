from moe.base.mixture import BaseMoE
from moe.base.strategies import BaseExpertStrategy


class QueryAugmentationStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['expert_input'],
            'topic': state['kwargs'].get('topic', 'No specific topic, go ahead and imply it as best as you can from the query'),
        })
        state['expert_output'] = "Successfully enhanced the queries."
        state['kwargs']['enhanced_queries'] = [enhancement['query'] for enhancement in output['queries']]
        state['next'] = 'HydeExpert'
        return state


class HydeStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        outputs = []
        for q_ in state['kwargs']['enhanced_queries']:
            local_output = expert.invoke({
                'input': q_,
                'topic': state['kwargs'].get('topic', 'No specific topic, imply it as best as you can from the query'),
                'context':  state['kwargs'].get('context', ''),
            })
            outputs.append(local_output)
        state['expert_output'] = "Successfully generated hypothetical documents."
        state['kwargs']['hyde'] = outputs
        state['next'] = BaseMoE.FINISH
        return state
