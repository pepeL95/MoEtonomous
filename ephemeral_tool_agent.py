from typing import List
from langchain_community.llms import BaseLLM
from james_bond.agents.agent import Agent
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent

import time
import random

class EphemeralToolAgent(Agent):   
    """Tool calling agent without memory."""
    def __init__(
            self, 
            name:str, 
            llm:BaseLLM, 
            system_prompt:str, 
            tools: List[BaseTool], 
            prompt_template:str = None, 
            verbose:bool = False, 
            parser:BaseOutputParser = None, 
            is_silent_caller:bool = True
            ) -> None:
        
        super().__init__(name, llm=llm, system_prompt=system_prompt, prompt_template=prompt_template, parser=parser)
        self._tools = tools
        self._is_silent_caller = is_silent_caller
        self._verbose = verbose
        # Init agent
        self._make_agent()

################################################### GETTERS #####################################################
    
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

######################################## PRIVATE METHODS #########################################################

    def _make_agent(self):
        """Create a tool_calling_agent using Langchain."""
        if self._prompt_template is None and self._system_prompt is None:
            raise ValueError("Must have at least one of Union[system_prompt, prompt_template].")
        
        # Build prompt
        human_template = ("human", "{input}")
        if self._prompt_template is not None and self._system_prompt is not None:
            human_template = ('human', self._prompt_template)
        
        elif self._prompt_template is not None and self._system_prompt is None:
            prompt = ChatPromptTemplate.from_messages([
                self._prompt_template,
                ("placeholder", "{agent_scratchpad}"),
                ])
            
            if self._is_silent_caller:
                self._agent = create_tool_calling_agent(self._llm, self._tools, prompt) | self._identity_fn
            else:
                self._agent = create_tool_calling_agent(self._llm, self._tools, prompt)
            return
        
        prompt = ChatPromptTemplate.from_messages([
                ("system", self._system_prompt),
                human_template,
                ("placeholder", "{agent_scratchpad}"),
                ])
        
        if self._is_silent_caller:
            self._agent = create_tool_calling_agent(self._llm, self._tools, prompt) | self._identity_fn
        else:
            self._agent = create_tool_calling_agent(self._llm, self._tools, prompt)

    @classmethod
    def _identity_fn(cls, data):
        """This makes sure to just return the tool outputs directly. Equivalent to tools return_direct."""
        return data

    @classmethod
    def _stream_results(cls, input_string):
        '''
        Temporary streaming, while we figure out how in the heck to actually do it.
        '''
        length = len(input_string)
        i = 0
        while i < length:
            chunk_size = random.randint(1, 5)
            chunk = input_string[i:i + chunk_size]
            print(chunk, end='', flush=True)
            i += chunk_size
            time.sleep(random.uniform(0.005, 0.01))
        print()
        print()

############################################# PUBLIC METHODS ####################################################

    def invoke(self, input: str | dict):
        """Invoke agentic chain."""
        if not any([isinstance(input, str), isinstance(input, dict)]):
            raise ValueError(f'Input must be one of Union[str, dict]. Got {type(input)}')
        
        agent_executor = AgentExecutor(agent=self._agent, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
        if self._parser is not None:
            agent_executor = (
                agent_executor
                | RunnableLambda(lambda response: response["output"]) | self._parser
            )
        
        return agent_executor.invoke(input)
    
    def stream(self, input):
        '''Stream agent response'''
        result = self.invoke(input)
        if isinstance(result["output"], str):
            # TODO: Actual streaming
            self._stream_results(result["output"])
        else:
            # No can do: streaming is for strings, loser
            print(result["output"])
        return result

    def get_chain(self):
        return super().get_chain()
