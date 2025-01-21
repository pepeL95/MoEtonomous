from abc import abstractmethod


class Strategy:
    @abstractmethod
    def execute(self, moe, input):
        pass

class DefaultStrategy(Strategy):
    def execute(self, moe, input):
        return moe.invoke(input)