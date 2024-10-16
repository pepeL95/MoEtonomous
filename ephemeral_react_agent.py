from james_bond.agents.agent import Agent

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
        '''Create a react_agent using Langchain.'''
        if self._system_prompt is None and self._prompt_template is None:
            prompt = hub.pull("hwchase17/react")
            self._agent = create_react_agent(self._llm, self._tools, prompt)
            return
        
        if self._prompt_template is not None:
            prompt = PromptTemplate.from_template(self._prompt_template)
            self._agent = create_react_agent(self._llm, self._tools, prompt)
            return
        
        # Get template
        human_template = ("human", hub.pull("hwchase17/react").template)
        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
                ("system", self._system_prompt),
                human_template
                ])
        # Build agent
        self._agent = create_react_agent(self._llm, self._tools, prompt)

############################################# PUBLIC METHODS ####################################################

    def get_chain(self):
        return super().get_chain()
    
    def invoke(self, input_object):
        """Invoke agentic chain."""
        agent_executor = AgentExecutor(agent=self._agent, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
        if self._parser is not None:
            agent_executor = (
                agent_executor
                | RunnableLambda(lambda response: response["output"]) | self._parser
            )
        return agent_executor.invoke(input_object)
