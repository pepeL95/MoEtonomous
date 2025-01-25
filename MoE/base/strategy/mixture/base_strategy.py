from abc import abstractmethod


class MoEStrategy:
    @abstractmethod
    def execute(self, moe, input):
        pass
