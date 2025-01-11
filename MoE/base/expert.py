from JB007.base.agent import Agent
from langchain_core.runnables import Runnable

class Expert:
    def __init__(self, agent:Agent | Runnable, description:str, name:str) -> None:
        self.name = name
        self.description = description
        self.agent = agent

    def invoke(self, input):
        return self.agent.invoke(input)
