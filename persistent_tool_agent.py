
from james_bond.agents.ephemeral_tool_agent import EphemeralToolAgent

from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain.agents import AgentExecutor
from typing import List

class PersistentToolAgent(EphemeralToolAgent):
    def __init__(self, name, llm, tools, system_prompt=None, prompt_template=None, verbose=False, parser=None, is_silent_caller=True):
        super().__init__(name, llm=llm, system_prompt=system_prompt, tools=tools, prompt_template=prompt_template, verbose=verbose, parser=parser, is_silent_caller=is_silent_caller)

    def _make_agent(self):
        """Create a tool_calling_agent using Langchain."""
        if self._prompt_template is None and self._system_prompt is None:
            raise ValueError("Must have at least one of Union[system_prompt, prompt_template].")
        
        # Build prompt
        human_template = ("human", "{input}")
        if self._prompt_template is not None and self._system_prompt is not None:
            human_template = ("human", self._prompt_template)
        
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
                ("placeholder", "{chat_history}"),
                human_template,
                ("placeholder", "{agent_scratchpad}"),
                ])
        
        if self._is_silent_caller:
            self._agent = create_tool_calling_agent(self._llm, self._tools, prompt) | self._identity_fn
        else:
            self._agent = create_tool_calling_agent(self._llm, self._tools, prompt)

    def invoke(self, input: str | dict | List[BaseMessage]):
        """Invoke agentic chain."""
        if any([isinstance(input, str), isinstance(input, dict)]):
            input_object = input
        
        elif isinstance(input, list) and all([isinstance(msg, BaseMessage) for msg in input]):
            input_object = {'chat_history': input[:-1], 'input': input[-1]}
        
        else:
            raise ValueError(f'Input must be one of Union[str, dict, List[BaseMessage]]. Got {type(input)}')

        agent_executor = AgentExecutor(agent=self._agent, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
        if self._parser is not None:
            agent_executor = (
                agent_executor
                | RunnableLambda(lambda response: response["output"]) | self._parser
            )
        
        ret = agent_executor.invoke(input_object)
        return ret
        
    def get_chain(self):
        ret = super().get_chain()
        return ret

    def stream(self, input_object):
        ret = super().stream(input_object)
        return ret
    