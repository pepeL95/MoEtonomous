from abc import abstractmethod

class BaseExpertStrategy:
    def __init__(self, next=None):
        self.next = next

    @abstractmethod
    def execute(self, expert, state):
        raise NotImplementedError('Expert strategy needs to be implemented.')

class BaseMoEStrategy:
    def __init__(self, next=None):
        self.next = next
        
    @abstractmethod
    def execute(self, moe, input):
        raise NotImplementedError('MoE strategy needs to be implemented.')
