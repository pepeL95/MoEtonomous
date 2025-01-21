from dotenv import load_dotenv
import sys
import os


if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"]) # .env file path
    sys.path.append(os.environ.get('SRC'))

from MoE.mixtures.raggaeton.pretrieval_MoE import PretrievalMoE
from dev_tools.utils.clifont import print_cli_message
from MoE.xperts.expert_factory import ExpertFactory
from MoE.config.debug import Debug
from MoE.base.expert.lazy_expert import LazyExpert

from dev_tools.enums.llms import LLMs

from langchain_core.runnables import RunnableLambda


class PreRetrievalMoE:
    @staticmethod
    def get():
        
        # Init experts
        _router = LazyExpert(
            name='PreRetrievalOrchestrator', 
            description='Orchestrates the RAG experts in the pre-retrieval step in a modular RAG pipeline', 
            agent=RunnableLambda(lambda state: (
                f'\nAction: {ExpertFactory.Directory.QueryXtractionXpert}\n'
                f'Action Input: {state['input']}\n'
            )))
        
        query_xxpert = ExpertFactory.get(
            xpert=ExpertFactory.Directory.QueryXtractionXpert,
            llm=LLMs.Gemini()
        )
        
        hyde_xpert = ExpertFactory.get(
            xpert=ExpertFactory.Directory.HyDExpert,
            llm=LLMs.Gemini()
        )

        # Init MoE
        pretrievalMoE = PretrievalMoE(
            name='PretrievalMoE',
            description=(
                'A master at orchestrating the pre-retrieval step of a Retrieval Augmented Generation (RAG) pipeline. '
                'It returns a hypothetical answer that must be given to the PostrievalMoE.'
                'Use this expert at the beginning of the pipeline. '
            ),
            router=_router,
            experts=[query_xxpert, hyde_xpert],
            verbose=Debug.Verbosity.low,
        ).build()
        
        return pretrievalMoE


################################## REGRESSION TEST #########################################

if __name__ == '__main__':
    # Run
    pre_retrievalMoE = PreRetrievalMoE.get()
    user_input = input('Enter query: ')
    state = pre_retrievalMoE.invoke({'input': user_input})
    print_cli_message('**assistant: **')
    print(state['expert_output'])