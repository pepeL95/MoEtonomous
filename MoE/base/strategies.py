from abc import abstractmethod

class BaseExpertStrategy:
    @abstractmethod
    def execute(self, expert, state):
        raise NotImplementedError('Expert strategy needs to be implemented.')

class BaseMoEStrategy:
    @abstractmethod
    def execute(self, moe, input):
        raise NotImplementedError('MoE strategy needs to be implemented.')
