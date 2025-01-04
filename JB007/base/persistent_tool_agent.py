
from JB007.base.ephemeral_tool_agent import EphemeralToolAgent

from typing import List

from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain_core.runnables import RunnablePassthrough

from JB007.parsers.prompt import IdentityPromptParser

class PersistentToolAgent(EphemeralToolAgent):
    def __init__(
            self,
            name,
            llm,
            tools,
            system_prompt=None,
            prompt_template=None,
            verbose=False,
            prompt_parser=IdentityPromptParser(),
            output_parser=None,
            is_silent_caller=True):
        
        super().__init__(name, llm=llm, system_prompt=system_prompt, tools=tools, prompt_template=prompt_template, verbose=verbose, prompt_parser=prompt_parser, output_parser=output_parser, is_silent_caller=is_silent_caller)

######################################## PRIVATE METHODS #########################################################

    def _make_agent(self):
        """Create a tool_calling_agent using Langchain."""
        if self._prompt_template is None and self._system_prompt is None:
            raise ValueError("Must have at least one of Union[system_prompt, prompt_template].")
        
        # Parse prompts
        sys_prompt = self._system_prompt and self.prompt_parser.parseSys(self._system_prompt)
        usr_prompt = self.prompt_parser.parseUser(self._prompt_template if self.prompt_template else "{input}")

        human_template = ("human", usr_prompt)
        if self._system_prompt is None:
            prompt = ChatPromptTemplate.from_messages([
                ("placeholder", "{chat_history}"),
                human_template,
                ("placeholder", "{agent_scratchpad}"),
                ])
        
        if self._system_prompt is not None:            
            prompt = ChatPromptTemplate.from_messages([
                    ("system", sys_prompt),
                    ("placeholder", "{chat_history}"),
                    human_template,
                    ("placeholder", "{agent_scratchpad}"),
                    ])
            
        if self._is_silent_caller:
            self._agent = create_tool_calling_agent(self._llm, self._tools, prompt) | self._identity_fn
        else:
            self._agent = create_tool_calling_agent(self._llm, self._tools, prompt)


    def _invoke_with_prompt_template(self, input, config: RunnableConfig | None = None, stream:bool=False):
        if any([isinstance(input, str), isinstance(input, dict)]):
            input_object = input
        
        elif isinstance(input, list) and all([isinstance(msg, BaseMessage) for msg in input]):
            input_object = {'chat_history': input[:-1], 'input': input[-1]}
        
        else:
            raise ValueError(f'Input must be one of Union[str, dict, List[BaseMessage]]. Got {type(input)}')

        # To parse, or not to parse, that is the question
        if self._output_parser is None:
            self._output_parser = RunnablePassthrough()

        agent_executor = (
            AgentExecutor(agent=self._agent, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
            | RunnableLambda(lambda response: response["output"]) 
            | self._output_parser
        )
        
        # To stream, or not to stream, that is the question
        if stream:
            ret = agent_executor.stream(input_object, config)
            return ret
        ret = agent_executor.invoke(input_object, config)
        return ret
    

    def _invoke_without_prompt_template(self, input, config: RunnableConfig | None = None, stream:bool=False):
        return self._invoke_with_prompt_template(input, config, stream)


######################################## PUBLIC METHODS #########################################################

    def invoke(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: RunnableConfig | None = None):
        return super().invoke(input, config)

    def stream(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: RunnableConfig | None = None):
        return super().stream(input, config)
    
    def get_chain(self):
        return super().get_chain()