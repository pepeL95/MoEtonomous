from langchain_core.messages import BaseMessage
from JB007.base.agent import Agent

from typing import List
from langchain import hub
from langchain_core.tools import BaseTool
from langchain_community.llms import BaseLLM
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import BaseOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

class EphemeralReactAgent(Agent):
    """React agent without memory."""
    def __init__(
            self, 
            name: str, 
            llm: BaseLLM, 
            tools: List[BaseTool],
            system_prompt = None,
            prompt_template: str = None, 
            parser:BaseOutputParser = None,
            verbose:bool = False
            ) -> None:
        
        super().__init__(name, llm=llm, prompt_template=prompt_template, system_prompt=system_prompt, parser=parser)
        self._tools = tools
        self._verbose = verbose
        # Init agent
        self._make_agent()

######################################## PRIVATE METHODS #########################################################

    def _make_agent(self):
        '''Create a react_agent using Langchain.'''
        if self._system_prompt is None:
            prompt = self._prompt_template or hub.pull("hwchase17/react")
            self._agent = create_react_agent(self._llm, self._tools, prompt)
            return
        
        if self._system_prompt is not None:
            # Get template
            human_template = ("human", self._prompt_template or hub.pull("hwchase17/react").template)
            # Build prompt
            prompt = ChatPromptTemplate.from_messages([
                    ("system", self._system_prompt),
                    human_template
                    ])
            # Build agent
            self._agent = create_react_agent(self._llm, self._tools, prompt)
            return

    def _invoke_with_prompt_template(self, input, stream):
        agent_executor = AgentExecutor(agent=self._agent, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
        if self._parser is not None:
            agent_executor = (
                agent_executor
                | RunnableLambda(lambda response: response["output"]) | self._parser
            )
        return agent_executor.invoke(input)
    
    def _invoke_without_prompt_template(self, input, stream):
        # There is always a prompt template, by design
        return self._invoke_with_prompt_template(input, stream)

############################################# PUBLIC METHODS ####################################################

    def get_chain(self):
        return super().get_chain()
    
    def invoke(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage]):
        return super().invoke(input)
    
    def stream(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage]):
        return super().stream(input)
