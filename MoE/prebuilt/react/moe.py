############################# SYS PATH LOAD ################################

from dotenv import load_dotenv
import sys
import os

from dev_tools.enums.llms import LLMs
from dev_tools.enums.prompt_parsers import PromptParsers

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"])  # .env file path
    sys.path.append(os.environ.get('SRC'))

###########################################################################

from agents.config.debug import Debug
from dev_tools.enums.llms import LLMs
from MoE.base.mixture.base_mixture import MoEBuilder
from dev_tools.enums.prompt_parsers import PromptParsers
from MoE.mixtures.react.experts.factory import ReActFactory, ReActDirectory


if __name__ == '__main__':
    # Init ReAct MoE
    react = MoEBuilder()\
        .set_name('ReActMoE')\
        .set_description('MoE that implements the ReAct framework for LLMs. It thinks, plans, and acts to non-naively fulfill a request.')\
        .set_router(ReActFactory.get(expert_name=ReActDirectory.Router))\
        .set_verbosity(Debug.Verbosity.low)\
        .set_experts([
            ReActFactory.get(expert_name=ReActDirectory.IntentXtractor,
                             llm=LLMs.Phi35(), prompt_parser=PromptParsers.Phi35()),
            ReActFactory.get(expert_name=ReActDirectory.PlanningXpert, llm=LLMs.Gemini(
            ), prompt_parser=PromptParsers.Identity()),
            ReActFactory.get(expert_name=ReActDirectory.SynthesisXpert,
                             llm=LLMs.Phi35(), prompt_parser=PromptParsers.Phi35()),
        ])\
        .build()

    # Run loop
    while user_input := input('Enter your prompt here: '):
        state = react.invoke({
            'input': user_input,
            'expert_input': user_input,
            'next': ReActDirectory.PlanningXpert,
            'kwargs': {
                'scratchpad': '',
                'previous_expert': None,
                'experts': '{GeneralKnowledgeExpert: Use it as a general purpose expert, WebSearchExpert: Use it for searching the web}',
                'expert_names': ['GeneralKnowledgeExpert', 'WebSearchExpert'],
            },
        })

        print(state['expert_output'])
