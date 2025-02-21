from moe.base.strategies import BaseExpertStrategy


class GenXpertStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['expert_input'],
            'context': state['ephemeral_mem'].messages[-5:] or '',
        })
        state['expert_output'] = output
        return state


class WebSearchStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['expert_input'],
            'context': state['ephemeral_mem'].messages[-5:]
        })
        state['expert_output'] = output
        return state
