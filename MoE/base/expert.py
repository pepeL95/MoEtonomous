from james_bond.agents.agent import Agent
from typing import Any

class Expert:
    def __init__(self, agent:Agent | Any, description:str, name: str=None) -> None:
        self.name = name or agent.name
        self.description = description
        self.agent = agent

    def invoke(self, input):
        return self.agent.invoke(input)
