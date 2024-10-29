from JB007.base.agent import Agent

from typing import Optional
from langchain_core.runnables import RunnableLambda, RunnableConfig

class Router:
    '''ReAct routing expert'''
    def __init__(self, description:str, name: str, agent: Agent | RunnableLambda) -> None:
        self.name = name
        self.description = description
        self.agent = agent
        if isinstance(self.agent, Agent):    
            stop = ['\nExpert Response:']
            self.agent.llm = self.agent.llm.bind(stop=stop)

    def invoke(self, input, config:Optional[RunnableConfig]=None):
        return self.agent.invoke(input=input, config=config)