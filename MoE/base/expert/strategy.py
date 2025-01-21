from abc import abstractmethod


class Strategy:
    @abstractmethod
    def execute(self, expert, state):
        pass

class DefaultStrategy(Strategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['input']
        })

        state['expert_output'] = output
        state['next'] = 'Router'
        return state