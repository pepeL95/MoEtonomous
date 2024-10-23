from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"]) # .env file path
    sys.path.append(os.environ.get('SRC'))


from MoE.base.router import Router
from MoE.mixtures.raggaeton.modular_MoE import ModularRAGMoE
from MoE.xperts.expert_factory import ExpertFactory
from MoE.config.debug import Debug

from dev.tests.regression.MoEs.rag.post import PostRetrievalMoE
from dev.tests.regression.MoEs.rag.pre import PreRetrievalMoE

from dev_tools.utils.clifont import print_cli_message
from dev_tools.enums.embeddings import Embeddings
from dev_tools.enums.llms import LLMs

from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

class ModularRagMoE:
    @staticmethod
    def get():  
        # Init experts
        router = Router(
            name='RAGOrchestrator',
            description=None,
            agent=RunnableLambda(lambda state: (
                f'\nAction: PretrievalMoE\n'
                f'Action Input: {state['input']}\n'
            ))
        )
        retriever_xpert = ExpertFactory.get(
            xpert=ExpertFactory.Directory.RetrieverExpert, 
            llm=None, 
            retriever=Chroma(
                    collection_name='toy-embeddings', 
                    persist_directory=os.getenv('VDB_PATH'), 
                    embedding_function=Embeddings.sentence_transformers_mpnet(),
            ).as_retriever(search_kwargs={'k': 10})
        )
        pretrievalMoE = PreRetrievalMoE.get()
        postrievalMoE = PostRetrievalMoE.get()  

        # Init MoE
        MoE = ModularRAGMoE(
            name='ModularRAGMoE',
            description=None,
            experts=[pretrievalMoE, retriever_xpert, postrievalMoE],
            router=router,
            verbose=Debug.Verbosity.low,
        ).build_MoE()

        return MoE


################################## REGRESSION TEST #########################################


if __name__ == '__main__':
    # Run
    ragMoE = ModularRagMoE.get()
    user_input = input('Enter query: ')
    state = ragMoE.invoke({'input': user_input, 'kwargs': {'topic': 'AWS Lambda'}})
    print_cli_message('**assistant: **' + state['expert_output'])