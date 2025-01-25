from MoE.base.strategy.mixture.base_strategy import MoEStrategy


class DefaultMoEStrategy(MoEStrategy):
    def execute(self, moe, input):
        output = moe.invoke(input)
        return output