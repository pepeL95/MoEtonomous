from moe.base.mixture import BaseMoE
from moe.base.expert import BaseExpert
from moe.base.strategies import BaseExpertStrategy, BaseMoEStrategy

class RouterStrategy(BaseMoEStrategy):
    '''Strategy for the MoE itself'''
    def execute(self, moe, input):
        output = moe.invoke({
            'input': input['input'],
            'kwargs': {
                'scratchpad': input['scratchpad'],
                'previous_expert': input['previous_expert'],
                'experts': input['experts'],
                'expert_names': input['expert_names'],
            }
        })

        return output

class InnerStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        # Note: state = input in this case by design
        output = expert.invoke(state)
        state = {'expert_output': output}
        return state


class IntentXtractStrategy(BaseExpertStrategy):
    def execute(cls, expert: BaseExpert, state: BaseMoE.State):
        output = expert.invoke({
            'input': state['expert_input'],
            'context': state['ephemeral_mem'].messages[-5:] or '',
        })

        state['next'] = 'PlanningXpert'
        state['expert_output'] = output
        return state


class PlanningStrategy(BaseExpertStrategy):
    def execute(cls, expert: BaseExpert, state: BaseMoE.State):
        output = expert.invoke({
            'input': state['expert_input'],
            'experts': state['kwargs']['experts'],
            'expert_names': state['kwargs']['expert_names'],
            'scratchpad': state['kwargs']['scratchpad'] or '',
        })

        state['next'] = 'SynthesisXpert'
        state['expert_output'] = output  # Plan + Action + Action Input
        return state


class SynthesisStrategy(BaseExpertStrategy):
    def execute(cls, expert: BaseExpert, state: BaseMoE.State):
        prev_scratchpad, _, prev_xpert_response = state['kwargs']['scratchpad'].rpartition(
            f'{state['kwargs']['previous_expert']} Response: ')
        if prev_scratchpad and prev_xpert_response:
            compressed_xpert_response = expert.invoke(
                {'input': prev_xpert_response})
            # Context compression strategy
            state['router_scratchpad'] = prev_scratchpad + \
                f'{state['kwargs']['previous_expert']} Response: ' + \
                compressed_xpert_response

        # state['next'] = 'PlanExecutor'
        state['next'] = BaseMoE.FINISH
        return state
