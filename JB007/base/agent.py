from abc import abstractmethod
from typing import List, Optional

from langchain_community.llms import BaseLLM
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser

from JB007.parsers.prompt import BasePromptParser, IdentityPromptParser

class Agent(Runnable):
    """Abstract class that offers a basic interface for specific agents"""
    def __init__(
            self,
            name:str,
            llm:BaseLLM = None,
            system_prompt:str = None,
            prompt_template:str = None,
            prompt_parser:BasePromptParser = IdentityPromptParser(),
            output_parser:BaseOutputParser = StrOutputParser()
        ) -> None:
        
        self._name = name
        self._llm = llm
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template
        self._prompt_parser = prompt_parser
        self._output_parser = output_parser
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
    def output_parser(self):
        return self._output_parser
    
    @property
    def prompt_parser(self):
        return self._prompt_parser
    
    @property
    def name(self):
        return self._name
    
################################################## SETTERS #####################################################

    @llm.setter
    def llm(self, llm:BaseLLM):
        if not isinstance(llm, BaseLLM):
            raise TypeError(f'llm must be of type BaseLLM. Got {type(llm)}')
        
        self._llm = llm
        self._make_agent()

    @system_prompt.setter
    def system_prompt(self, system_prompt:str):
        if not isinstance(system_prompt, str):
            raise TypeError(f'system_prompt must be of type str. Got {type(system_prompt)}')
        
        self._system_prompt = self._prompt_parser.parseSys(system_prompt)
        self._make_agent()

    @prompt_template.setter
    def prompt_template(self, prompt_template:str):
        if not isinstance(prompt_template, str):
            raise TypeError(f'prompt_template must be of type str. Got {type(prompt_template)}')
        
        self._prompt_template = self._prompt_parser.parseUser(prompt_template)
        self._make_agent()

    @prompt_parser.setter
    def prompt_parser(self, prompt_parser:BasePromptParser):
        if not isinstance(prompt_parser, BasePromptParser):
            raise TypeError(f'prompt_parser must be of type BasePromptParserGot {type(prompt_parser)}')

        self._prompt_parser = prompt_parser
        self._make_agent()

    @output_parser.setter
    def output_parser(self, output_parser:BaseOutputParser):
        if not isinstance(output_parser, BaseOutputParser):
            raise TypeError(f'output_parser must be of type BaseOutputParser. Got {type(output_parser)}')
        
        self._output_parser = output_parser
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
            return self._invoke_without_prompt_template(input, config, stream=False)
        return self._invoke_with_prompt_template(input, config, stream=False)
    
    def stream(self, input:str | dict | List[dict] | BaseMessage | List[BaseMessage], config: Optional[RunnableConfig] = None):
        if not self._prompt_template:
            return self._invoke_without_prompt_template(input, config, stream=True)
        return self._invoke_with_prompt_template(input, config, stream=True)