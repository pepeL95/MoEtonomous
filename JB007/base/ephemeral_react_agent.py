from langchain_core.messages import BaseMessage
from JB007.base.agent import Agent

from typing import List
from langchain import hub
from langchain_core.tools import BaseTool
from langchain_community.llms import BaseLLM
from langchain_core.output_parsers import BaseOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough

from JB007.parsers.prompt import BasePromptParser, IdentityPromptParser

class EphemeralReactAgent(Agent):
    """React agent without memory."""
    def __init__(
            self, 
            name: str, 
            llm: BaseLLM, 
            tools: List[BaseTool],
            system_prompt = None,
            prompt_template: str = None, 
            prompt_parser:BasePromptParser = IdentityPromptParser(),
            output_parser:BaseOutputParser = None,
            verbose:bool = False
            ) -> None:
        
        super().__init__(name, llm=llm, prompt_template=prompt_template, system_prompt=system_prompt, prompt_parser=prompt_parser, output_parser=output_parser)
        self._tools = tools
        self._verbose = verbose
        # Init agent
        self._make_agent()

######################################## PRIVATE METHODS #########################################################

    def _make_agent(self):
        '''Create a react_agent using Langchain.'''
        # To parse, or not to parse, that is the question
        if self._prompt_parser is None:
            self.prompt_parser = IdentityPromptParser()
            
        if self._output_parser is None:
            self._output_parser = RunnablePassthrough()
            
        # Parse prompts
        sys_prompt = self._system_prompt and self.prompt_parser.parseSys(self._system_prompt)
        usr_prompt = self.prompt_parser.parseUser(self._prompt_template or hub.pull("hwchase17/react").template)

        if self._system_prompt is None:
            prompt = PromptTemplate.from_template(usr_prompt)
            self._agent = create_react_agent(self._llm, self._tools, prompt)
            return
        
        if self._system_prompt is not None:
            # Get template
            human_template = ("human", usr_prompt)
            # Build prompt
            prompt = ChatPromptTemplate.from_messages([
                    ("system", sys_prompt),
                    human_template
                    ])
            # Build agent
            self._agent = create_react_agent(self._llm, self._tools, prompt)
            return

    def _invoke_with_prompt_template(self, input, config=None, stream=False):
        agent_executor = (
            AgentExecutor(agent=self._agent, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
            | RunnableLambda(lambda response: response["output"])
            | self._output_parser
        )

        if stream:
            return agent_executor.stream(input, config)
        return agent_executor.invoke(input, config)
    
    def _invoke_without_prompt_template(self, input, config=None, stream=False):
        # There is always a prompt template, by design
        return self._invoke_with_prompt_template(input, config, stream)

############################################# PUBLIC METHODS ####################################################

    def get_chain(self):
        return super().get_chain()
    
    def invoke(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: RunnableConfig | None = None):
        return super().invoke(input, config)
    
    def stream(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: RunnableConfig | None = None):
        return super().stream(input, config)
