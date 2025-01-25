from abc import abstractmethod


class BaseExpertStrategy:
    @abstractmethod
    def execute(self, expert, state):
        pass
