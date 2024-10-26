from abc import abstractmethod
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

class Agent(Runnable):
    """Abstract class that offers a basic interface for specific agents"""
    def __init__(self, name, llm=None, system_prompt=None, prompt_template=None, parser=None):
        self._name = name
        self._llm = llm
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template
        self._parser = parser
        self._agent = None

################################################## GETTERS #####################################################

    @property
    def llm(self):
        return self._llm

    @property
    def prompt_template(self):
        return self._prompt_template
    
    @property
    def system_prompt(self):
        return self._system_prompt
    
    @property
    def parser(self):
        return self._parser
    
    @property
    def name(self):
        return self._name
    
################################################## SETTERS #####################################################

    @llm.setter
    def llm(self, llm):
        self._llm = llm
        self._make_agent()

    @prompt_template.setter
    def prompt_template(self, prompt_template):
        self._prompt_template = prompt_template
        self._make_agent()
    
    @system_prompt.setter
    def system_prompt(self, system_prompt):
        self._system_prompt = system_prompt
        self._make_agent()

    @parser.setter
    def parser(self, parser):
        self._parser = parser
        self._make_agent()

################################################## ABSTRACT #####################################################

    @abstractmethod
    def _make_agent(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    @abstractmethod
    def _invoke_with_prompt_template(self, input, stream):
        raise NotImplementedError("Subclass must implement abstract method")
    
    @abstractmethod
    def _invoke_without_prompt_template(self, input, stream):
        raise NotImplementedError("Subclass must implement abstract method")
    
################################################## CONCRETE #####################################################

    def get_chain(self):
        return self._agent
    
    def invoke(self, input:str | dict | List[dict] | BaseMessage | List[BaseMessage] , config: Optional[RunnableConfig] = None):
        if not self._prompt_template:
            return self._invoke_without_prompt_template(input, stream=False)
        return self._invoke_with_prompt_template(input, config, stream=False)
    
    def stream(self, input:str | dict | List[dict] | BaseMessage | List[BaseMessage], config: Optional[RunnableConfig] = None):
        if not self._prompt_template:
            return self._invoke_without_prompt_template(input, config, stream=True)
        return self._invoke_with_prompt_template(input, config, stream=True)