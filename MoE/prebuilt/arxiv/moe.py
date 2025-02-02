############################# SYS PATH LOAD ################################

from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"])  # .env file path
    sys.path.append(os.environ.get('SRC'))

###########################################################################

from agents.config.debug import Debug
from dev_tools.enums.llms import LLMs
from MoE.base.mixture.base_mixture import MoEBuilder
from MoE.mixtures.arxiv.experts.factory import ArxivDirectory, ArxivFactory

if __name__ == '__main__':
    # Init Chat MoE
    chat = MoEBuilder()\
        .set_name('ArxivMoE')\
        .set_description(None)\
        .set_router(ArxivFactory.get(expert_name=ArxivDirectory.Router))\
        .set_verbosity(Debug.Verbosity.quiet)\
        .set_experts([
            ArxivFactory.get(expert_name=ArxivDirectory.QbuilderXpert),
            ArxivFactory.get(expert_name=ArxivDirectory.SearchXpert),
            ArxivFactory.get(expert_name=ArxivDirectory.SigmaXpert),
            ])\
        .build()

    # Run
    user_input = input('user: ')
    state = chat.invoke({
        'input': user_input,
    })

    print(state['expert_output'])
