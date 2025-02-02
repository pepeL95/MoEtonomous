from typing import Optional

from agents.base.agent import BaseAgent

from langchain_core.runnables import Runnable, RunnableConfig

from moe.base.strategies import BaseExpertStrategy


class BaseExpert:
    def __init__(self, agent: BaseAgent | Runnable, description: str, name: str, strategy: BaseExpertStrategy) -> None:
        self.name = name
        self.agent = agent
        self.strategy = strategy
        self.description = description

    def invoke(self, input, config: Optional[RunnableConfig] = None):
        return self.agent.invoke(input, config=config)

    def execute_strategy(self, state):
        return self.strategy.execute(expert=self, state=state)

__all__ = '''
BaseExpert
'''