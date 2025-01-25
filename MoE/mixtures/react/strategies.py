from MoE.base.expert.base_expert import Expert
from MoE.base.strategy.expert.base_strategy import BaseExpertStrategy
from MoE.base.mixture.base_mixture import MoE


class RouterStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        # Note: state = input in this case by design
        output = expert.invoke(state)
        state = {'expert_output': output}
        return state


class IntentXtractStrategy(BaseExpertStrategy):
    def execute(cls, expert: Expert, state: MoE.State):
        output = expert.invoke({
            'input': state['expert_input'],
            'context': state['ephemeral_mem'].messages[-5:] or '',
        })

        state['next'] = 'PlanningXpert'
        state['expert_output'] = output
        return state


class PlanningStrategy(BaseExpertStrategy):
    def execute(cls, expert: Expert, state: MoE.State):
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
    def execute(cls, expert: Expert, state: MoE.State):
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
        state['next'] = MoE.FINISH
        return state
