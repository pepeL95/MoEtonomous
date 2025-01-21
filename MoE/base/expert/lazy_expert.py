from typing import List

from JB007.base.agent import Agent

from MoE.base.expert.base_expert import Expert
from MoE.base.expert.strategy import Strategy

class LazyExpert(Expert):
    '''Expert that stops generating after a given token'''
    def __init__(self, description:str, name: str, agent: Agent, strategy:Strategy, stop_tokens:List[str]=None) -> None:
        super().__init__(agent=agent, description=description, name=name, strategy=strategy)
        if stop_tokens is not None:
            self._stop_tokens = stop_tokens
            self.bind()
    
    #################### NEW GETTERS #####################
    @property
    def stop_tokens(self):
        return self._stop_tokens
    
    #################### NEW SETTERS #####################

    @stop_tokens.setter
    def stop_tokens(self, stop_tokens):
        self._stop_tokens = stop_tokens
        self.bind()

    #################### NEW METHODS #####################

    def bind(self):
        if isinstance(self.agent, Agent):
            self.agent.llm = self.agent.llm.bind(stop=self.stop_tokens)