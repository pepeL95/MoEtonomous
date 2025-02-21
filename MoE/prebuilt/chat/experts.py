from agents.tools.toolbox import WebsearchToolbox
from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
from agents.prebuilt.ephemeral_tool_agent import EphemeralToolAgent

from moe.annotations.core import Expert
from moe.prebuilt.router.moe import AutonomousRouter
from moe.prebuilt.chat.strategies import GenXpertStrategy, WebSearchStrategy

from langchain_core.output_parsers import StrOutputParser

from dev_tools.enums.llms import LLMs
from dev_tools.enums.prompt_parsers import PromptParsers


@Expert(GenXpertStrategy)
class GenXpert:
    '''Excellent expert on a wide range of topics. Default to this expert when not sure which expert to use.'''
    agent = EphemeralNLPAgent(
        name='GenAgent',
        llm=LLMs.Gemini(),
        prompt_parser=PromptParsers.Identity(),
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
    )


@Expert(WebSearchStrategy)
class WebSearchXpert:
    '''Excels at searching the web for real-time information.'''
    agent = EphemeralToolAgent(
        name='DuckDuckGoAgent',
        llm=LLMs.Gemini(),
        prompt_parser=PromptParsers.Identity(),
        tools=[WebsearchToolbox.duck_duck_go_tool()],
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
    )

#############################################################################################################################

class Factory:
    class Dir:
        Router: str = 'Router'
        GenXpert: str = 'GenXpert'
        WebSearchXpert: str = 'WebSearchXpert'
    
    @staticmethod
    def get(expert_name: str):
        if expert_name == Factory.Dir.Router:
            return AutonomousRouter(llm=LLMs.Gemini()).build()
        if expert_name == Factory.Dir.GenXpert:
            return GenXpert()
        if expert_name == Factory.Dir.WebSearchXpert:
            return WebSearchXpert()
        raise ValueError(f'No expert by name {expert_name} exists.')
