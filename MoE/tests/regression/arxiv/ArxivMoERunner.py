from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"]) # .env file path
    sys.path.append(os.environ.get('SRC'))

from MoE.config.debug import Debug
from MoE.mixtures.arxiv.arxivMoE import ArxivMoE
from MoE.xperts.expert_factory import ExpertFactory

from dev_tools.enums.llms import LLMs
from dev_tools.utils.clifont import print_cli_message

class ArxivMoERunner:
    @staticmethod
    def get():
        # Create router & xperts
        router = ExpertFactory.get(ExpertFactory.Directory.Router, llm=LLMs.Gemini())
        query_xpert = ExpertFactory.get(ExpertFactory.Directory.ArxivQbuilderXpert, llm=LLMs.Gemini())
        search_xpert = ExpertFactory.get(ExpertFactory.Directory.ArxivSearchXpert, llm=LLMs.Gemini())
        sigma_xpert = ExpertFactory.get(ExpertFactory.Directory.ArxivSigmaXpert, llm=LLMs.Gemini())

        # Init & return MoE
        arxivMoE = ArxivMoE(
            name='ArxivMoE',
            description=None,
            router=router,
            experts=[query_xpert, search_xpert, sigma_xpert],
            verbose=Debug.Verbosity.low,
        ).build()

        return arxivMoE


################################## REGRESSION TEST #########################################


if __name__ == '__main__':
    # Run
    arxiv = ArxivMoERunner.get()
    user_input = input('Enter query: ')
    state = arxiv.invoke({
        'input': user_input, 
        'next': ExpertFactory.Directory.ArxivQbuilderXpert, # Forcing first expert to be this
        'expert_input': user_input, # Forcing first expert input to be this
    })
    print_cli_message('**assistant: **' + state['expert_output'])