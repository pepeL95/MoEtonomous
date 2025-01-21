from MoE.base.mixture.strategy import Strategy

class MoEStrategy(Strategy):
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

class GenXpertStategy(Strategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['expert_input'],
            'context': state['ephemeral_mem'].messages[-5:] or '',
        })

        state['next'] = 'Router'
        state['expert_output'] = output
        return state
    
class WebSearchStrategy(Strategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['expert_input'],
            'context': state['ephemeral_mem'].messages[-5:]
        })

        state['next'] = 'Router'
        state['expert_output'] = output
        return state