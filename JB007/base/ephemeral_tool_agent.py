from typing import List
from langchain_community.llms import BaseLLM
from langchain_core.messages import BaseMessage
from JB007.base.agent import Agent
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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

######################################## PRIVATE METHODS #########################################################

    def _make_agent(self):
        """Create a tool_calling_agent using Langchain."""
        if self._prompt_template is None and self._system_prompt is None:
            raise ValueError("Must have at least one of Union[system_prompt, prompt_template].")
        
        # Build prompt
        human_template = ("human", "{input}")
        if self._system_prompt is None:
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


    def _invoke_with_prompt_template(self, input, stream):
        if not any([isinstance(input, str), isinstance(input, dict)]):
            raise ValueError(f'Input must be one of Union[str, dict]. Got {type(input)}')
        
        agent_executor = AgentExecutor(agent=self._agent, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
        if self._parser is None:
            self.parser = RunnablePassthrough()
        
        agent_executor = (
                agent_executor
                | RunnableLambda(lambda response: response["output"]) 
                | self._parser
            )
        
        if stream:
            return agent_executor.stream(input)
        return agent_executor.invoke(input)
    
    def _invoke_without_prompt_template(self, input, stream):
        '''Always a prompt template, by design'''
        return self._invoke_with_prompt_template(input, stream)

    ######################################## CLASS METHODS #########################################################

    @classmethod
    def _identity_fn(cls, data):
        """This makes sure to just return the tool outputs directly. Equivalent to tools return_direct."""
        return data

    @classmethod
    def _stream_results(cls, input_string):
        '''
        Mock streaming: a temporary fix while we figure out how in the heck to actually do it.
        '''
        length = len(input_string)
        i = 0
        while i < length:
            chunk_size = random.randint(1, 5)
            chunk = input_string[i:i + chunk_size]
            yield chunk
            i += chunk_size
            time.sleep(random.uniform(0.005, 0.01))

############################################# PUBLIC METHODS ####################################################

    def invoke(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage]):
        return super().invoke(input)
        
    def stream(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage]):
        return super().stream(input)

    def get_chain(self):
        return super().get_chain()
