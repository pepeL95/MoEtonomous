############################# SYS PATH LOAD ################################

from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"])  # .env file path
    sys.path.append(os.environ.get('SRC'))

###########################################################################

from JB007.config.debug import Debug
from MoE.base.mixture.base_mixture import MoEBuilder
from MoE.base.strategy.mixture.default import DefaultMoEStrategy
from MoE.mixtures.ragentive.modular.experts.factory import RagDirectory, RagFactory

if __name__ == '__main__':
    # Init Chat MoE
    modular_rag = MoEBuilder()\
        .set_name('ModularRagMoE')\
        .set_description(None)\
        .set_router(RagFactory.get(expert_name=RagDirectory.Router))\
        .set_verbosity(Debug.Verbosity.quiet)\
        .set_strategy(DefaultMoEStrategy())\
        .set_experts([
            RagFactory.get(expert_name=RagDirectory.PretrievalMoE),
            RagFactory.get(expert_name=RagDirectory.Retrieval),
            RagFactory.get(expert_name=RagDirectory.PostrievalMoE),
        ])\
        .build()

    # Run
    user_input = input('user: ')
    state = modular_rag.invoke({
        'input': user_input,
        'kwargs': {
            'topic': 'AWS Lambda'
        }
    })

    print(state['expert_output'])
