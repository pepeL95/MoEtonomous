############################# SYS PATH LOAD ################################

from dotenv import load_dotenv
import sys
import os


if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"])  # .env file path
    sys.path.append(os.environ.get('SRC'))

###########################################################################

from moe.examples.arxiv.experts import Factory
from moe.annotations.core import MoE, ForceFirst 
from moe.default.strategies import DefaultMoEStrategy

if __name__ == '__main__':
    
    @MoE(DefaultMoEStrategy)
    @ForceFirst('QbuilderXpert')
    class ArxivMoE:
        '''This MoE fetches and summarizes articles from the Arxiv api, given a natural language query'''
        experts = [
            Factory.get(expert_name=Factory.Dir.QbuilderXpert),
            Factory.get(expert_name=Factory.Dir.SearchXpert),
            Factory.get(expert_name=Factory.Dir.SigmaXpert),
        ]

    # Run
    arxiv_moe = ArxivMoE()
    user_input = input('user: ')
    state = arxiv_moe.invoke({
        'input': user_input,
    })

    print(state['expert_output'])
