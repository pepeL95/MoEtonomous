from JB007.config.debug import Debug
from JB007.toolbox.toolbox import Toolbox
from JB007.base.ephemeral_nlp_agent import EphemeralNLPAgent
from JB007.base.ephemeral_tool_agent import EphemeralToolAgent

from MoE.base.mixture.base_mixture import MoEBuilder
from MoE.base.expert.base_expert import Expert
from MoE.mixtures.react.experts.factory import ReActDirectory, ReActFactory
from MoE.mixtures.chat.strategies import GenXpertStategy, WebSearchStrategy, MoEStrategy

from langchain_core.output_parsers import StrOutputParser

from dev_tools.enums.llms import LLMs


class Router:
    @staticmethod
    def get(llm):
        return MoEBuilder()\
        .set_name(Router.__name__)\
        .set_description('MoE that implements the ReAct framework for LLMs. It thinks, plans, and acts to non-naively fulfill a request.')\
        .set_router(ReActFactory.get(expert_name=ReActDirectory.Router))\
        .set_verbosity(Debug.Verbosity.quiet)\
        .set_strategy(MoEStrategy())\
        .set_experts([
            ReActFactory.get(expert_name=ReActDirectory.IntentXtractor), 
            ReActFactory.get(expert_name=ReActDirectory.PlanningXpert),
            ReActFactory.get(expert_name=ReActDirectory.SynthesisXpert),
            ])\
        .build()
    
class GenXpert:
    '''Excellent expert on a wide range of topics such as coding, math, history, an much more!!. Default to this expert when not sure which expert to use.'''
    @staticmethod
    def get(llm) -> Expert:
        return Expert(
            name=GenXpert.__name__,
            description=GenXpert.__doc__,
            strategy=GenXpertStategy(),
            agent=EphemeralNLPAgent(
                name='GenAgent',
                llm=llm,
                system_prompt=(
                    "## Instructions\n"
                    "You are a general knowledge expert who thrives in giving accurate information.\n"
                    "You are part of a conversation with other experts who, together, collaborate to fulfill a user request.\n"
                    "Your input is given from another expert who needs you to answer it.\n"
                    "You are chosen for a reason! Do not ask for clarifications.\n"
                    "Respond to your queries with brief, fact-based answers as best as you can\n"
                    "Format your response nicely, using markdown.\n\n"
                    "**Consider the following context (if any):**\n"
                    "{context}\n\n"
                ),
            ),
        )

class WebSearchXpert:
    '''Excels at searching the web for gathering up-to-date and real-time information.'''

    @staticmethod
    def get(llm) -> Expert:
        return Expert(
            name=WebSearchXpert.__name__,
            description=WebSearchXpert.__doc__,
            strategy=WebSearchStrategy(),
            agent=EphemeralToolAgent(
                name='DuckDuckGoAgent',
                llm=llm,
                tools=[Toolbox.Websearch.duck_duck_go_tool()],
                output_parser=StrOutputParser(),
                system_prompt=(
                    "## Instructions\n"
                    "You are an web search expert who gathers information based in a given query. Use the duck_duck_go_tool provided for searching the web.\n"
                    "You are part of a conversation with other experts who, together, collaborate to fulfill a request.\n"
                    "Your input is given from another expert who needs you to answer it.\n"
                    "You are chosen for a reason! Do not ask for clarifications.\n"
                    "Before responding, build a **highly detailed synthesis** of the results you obtained, including sources.\n\n"
                    "**Consider the following context (if any):**\n"
                    "{context}\n\n"
                ),
            ), 
        )

class ChatDirectory:
    Router:str = Router.__name__
    GenXpert:str = GenXpert.__name__
    WebSearchXpert:str = WebSearchXpert.__name__
    

class ChatFactory:
    @staticmethod
    def get(expert_name:str):
        if expert_name == ChatDirectory.Router:
            return Router.get(llm=None)
        if expert_name == ChatDirectory.GenXpert:
            return GenXpert.get(llm=LLMs.Gemini())
        if expert_name == ChatDirectory.WebSearchXpert:
            return WebSearchXpert.get(llm=LLMs.Gemini())
        
        raise ValueError(f'No expert by name {expert_name} exists.')