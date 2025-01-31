from JB007.config.debug import Debug
from JB007.toolbox.toolbox import Toolbox
from JB007.base.ephemeral_nlp_agent import EphemeralNLPAgent
from JB007.base.ephemeral_tool_agent import EphemeralToolAgent

from MoE.base.mixture.base_mixture import MoE
from MoE.base.expert.base_expert import Expert
from MoE.mixtures.react.experts.factory import ReActDirectory, ReActFactory


from langchain_core.output_parsers import StrOutputParser

from dev_tools.enums.llms import LLMs
from dev_tools.enums.prompt_parsers import PromptParsers


class Router(MoE):
    '''MoE that implements the ReAct framework for LLMs. It thinks, plans, and acts to non-naively fulfill a request.'''

    def __init__(self, name=None, router=None, experts=None, description=None, strategy=None, verbose=Debug.Verbosity.quiet):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or Router.__name__,
            description=description or Router.__doc__,
            router=router or ReActFactory.get(expert_name=ReActDirectory.Router),
            strategy=strategy,
            verbose=verbose,
            experts=experts or [
                ReActFactory.get(expert_name=ReActDirectory.IntentXtractor),
                ReActFactory.get(expert_name=ReActDirectory.PlanningXpert),
                ReActFactory.get(expert_name=ReActDirectory.SynthesisXpert),
            ],
        )


class GenXpert(Expert):
    '''Excellent expert on a wide range of topics such as coding, math, history, an much more!!. Default to this expert when not sure which expert to use.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or GenXpert.__name__,
            description=description or GenXpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
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
            ),
        )


class WebSearchXpert(Expert):
    '''Excels at searching the web for gathering up-to-date and real-time information.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None):
        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or WebSearchXpert.__name__,
            description=description or WebSearchXpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralToolAgent(
                name='DuckDuckGoAgent',
                llm=LLMs.Gemini(),
                prompt_parser=PromptParsers.Identity(),
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
