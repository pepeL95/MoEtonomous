from typing import Optional

from JB007.base.agent import Agent

from langchain_core.runnables import Runnable, RunnableConfig

from MoE.base.expert.strategy import Strategy


class Expert:
    def __init__(self, agent:Agent | Runnable, description:str, name:str, strategy:Strategy) -> None:
        self.name = name
        self.description = description
        self.agent = agent
        self.strategy = strategy

    def invoke(self, input, config: Optional[RunnableConfig] = None):
        return self.agent.invoke(input, config=config)

    def execute_strategy(self, state):
        return self.strategy.execute(expert=self, state=state)
            