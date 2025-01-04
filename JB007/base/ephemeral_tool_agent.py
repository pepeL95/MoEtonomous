import time
import random

from typing import List

from JB007.base.agent import Agent
from JB007.parsers.prompt import BasePromptParser, IdentityPromptParser

from langchain_core.tools import BaseTool
from langchain_community.llms import BaseLLM
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough


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
            prompt_parser:BasePromptParser = IdentityPromptParser(),
            output_parser:BaseOutputParser = None, 
            is_silent_caller:bool = True
            ) -> None:
        
        super().__init__(name, llm=llm, system_prompt=system_prompt, prompt_template=prompt_template, prompt_parser=prompt_parser, output_parser=output_parser)
        self._tools = tools
        self._is_silent_caller = is_silent_caller
        self._verbose = verbose
        # Init agent
        self._make_agent()

######################################## PRIVATE METHODS #########################################################

    def _make_agent(self):
        """Create a tool_calling_agent using Langchain."""
        # Sanity check
        if self._prompt_template is None and self._system_prompt is None:
            raise ValueError("Must have at least one of Union[system_prompt, prompt_template].")
        
        # To parse, or not to parse, that is the question
        if self._prompt_parser is None:
            self.prompt_parser = IdentityPromptParser()
            
        if self._output_parser is None:
            self.output_parser = RunnablePassthrough()

        # Parse prompts
        sys_prompt = self._system_prompt and self.prompt_parser.parseSys(self._system_prompt)
        usr_prompt = self._prompt_template and self.prompt_parser.parseUser(self._prompt_template)

        # Build prompt
        human_template = ("human", "{input}")
        if self._prompt_template is not None:
            human_template = ('human', usr_prompt)
        
        if self._system_prompt is None:
            prompt = ChatPromptTemplate.from_messages([
                human_template,
                ("placeholder", "{agent_scratchpad}"),
                ])
            
            if self._is_silent_caller:
                self._agent = create_tool_calling_agent(self._llm, self._tools, prompt) | self._identity_fn
            else:
                self._agent = create_tool_calling_agent(self._llm, self._tools, prompt)
            return
        
        prompt = ChatPromptTemplate.from_messages([
                ("system", sys_prompt),
                human_template,
                ("placeholder", "{agent_scratchpad}"),
                ])
        
        if self._is_silent_caller:
            self._agent = create_tool_calling_agent(self._llm, self._tools, prompt) | self._identity_fn
        else:
            self._agent = create_tool_calling_agent(self._llm, self._tools, prompt)


    def _invoke_with_prompt_template(self, input, config: RunnableConfig | None = None, stream:bool=False):
        if not any([isinstance(input, str), isinstance(input, dict)]):
            raise ValueError(f'Input must be one of Union[str, dict]. Got {type(input)}')
        
        agent_executor = (
            AgentExecutor(agent=self._agent, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
            | RunnableLambda(lambda response: response["output"]) 
            | self._output_parser
        )
        
        if stream:
            return agent_executor.stream(input, config)
        return agent_executor.invoke(input, config)
    
    def _invoke_without_prompt_template(self, input, config: RunnableConfig | None = None, stream:bool=False):
        '''Always a prompt template, by design'''
        return self._invoke_with_prompt_template(input, config, stream)

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

    def invoke(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: RunnableConfig | None = None):
        return super().invoke(input, config)
        
    def stream(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: RunnableConfig | None = None):
        return super().stream(input, config)

    def get_chain(self):
        return super().get_chain()
