from moe.base.strategies import BaseMoEStrategy, BaseExpertStrategy

class RouterStrategy(BaseMoEStrategy):
    def execute(self, moe, input):
        return moe.invoke({
            'input': input['input'],
            'kwargs': {
                'scratchpad': input['scratchpad'],
                'previous_expert': input['previous_expert'],
                'experts': input['experts'],
                'expert_names': input['expert_names'],
            },
        })


class GenXpertStategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['expert_input'],
            'context': state['ephemeral_mem'].messages[-5:] or '',
        })

        state['next'] = 'Router'
        state['expert_output'] = output
        return state


class WebSearchStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['expert_input'],
            'context': state['ephemeral_mem'].messages[-5:]
        })

        state['next'] = 'Router'
        state['expert_output'] = output
        return state
