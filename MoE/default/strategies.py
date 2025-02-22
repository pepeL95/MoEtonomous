from moe.base.strategies import BaseMoEStrategy, BaseExpertStrategy


class DefaultMoEStrategy(BaseMoEStrategy):
    def execute(self, moe, input):
        output = moe.invoke(input)
        return output

class DefaultExpertStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke(state)
        return {'expert_output': output}