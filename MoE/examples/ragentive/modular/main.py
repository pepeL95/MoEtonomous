############################# SYS PATH LOAD ################################

from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"])  # .env file path
    sys.path.append(os.environ.get('SRC'))

###########################################################################

from moe.annotations.core import ForceFirst, MoE
from moe.default.strategies import DefaultMoEStrategy
from moe.examples.ragentive.modular.experts import Factory

if __name__ == '__main__':
    # Define MoE
    @MoE(DefaultMoEStrategy)
    @ForceFirst('Pretrieval')
    class Ragentive:
        '''
        Sample implementation of a modular, agentive RAG pipeline.
        Note: Assumes data embeddings at RAG/data/vector/toy-embeddings
        '''
        experts = [
            Factory.get(expert_name=Factory.Dir.Pretrieval),
            Factory.get(expert_name=Factory.Dir.Retrieval),
            Factory.get(expert_name=Factory.Dir.Postrieval),
        ]

    # Run
    modular_rag = Ragentive()
    user_input = input('user: ')
    state = modular_rag.invoke({
        'input': user_input,
        'kwargs': {
            'topic': 'AWS Lambda'
        }
    })

    print(state['expert_output'])
