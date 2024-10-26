from langchain_core.runnables import Runnable

class Expert:
    def __init__(self, agent:Runnable, description:str, name:str) -> None:
        self.name = name
        self.description = description
        self.agent = agent

    def invoke(self, input):
        return self.agent.invoke(input)
