############################# SYS PATH LOAD ################################

from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"]) # .env file path
    sys.path.append(os.environ.get('SRC'))

###########################################################################

from JB007.config.debug import Debug
from MoE.base.mixture.base_mixture import MoEBuilder
from MoE.mixtures.react.experts.factory import ReActFactory, ReActDirectory


if __name__ == '__main__':
    react = MoEBuilder()\
        .set_name('ReActMoE')\
        .set_description('MoE that implements the ReAct framework for LLMs. It thinks, plans, and acts to non-naively fulfill a request.')\
        .set_router(ReActFactory.get(expert_name=ReActDirectory.Router))\
        .set_verbosity(Debug.Verbosity.low)\
        .set_experts([
            ReActFactory.get(expert_name=ReActDirectory.IntentXtractor), 
            ReActFactory.get(expert_name=ReActDirectory.PlanningXpert),
            ReActFactory.get(expert_name=ReActDirectory.SynthesisXpert),
            ])\
        .build()

    while user_input := input('Enter your prompt here: '):
        output_state = react.invoke({
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

        print(output_state['expert_output'])