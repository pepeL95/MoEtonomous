############################# SYS PATH LOAD ################################

from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"])  # .env file path
    sys.path.append(os.environ.get('SRC'))

###########################################################################

from moe.annotations.core import MoE, ForceFirst
from moe.default.strategies import DefaultMoEStrategy
from moe.examples.history_index.experts import IndexXpert, MetadataXpert, ResponseXpert, ThoughtXpert



def main():
    """Run Strategy"""
    @MoE(DefaultMoEStrategy)
    @ForceFirst('ThoughtXpert')
    class HistoryIndexMoE:
        '''Reasoning MoE'''
        experts = [
            ThoughtXpert(),
            ResponseXpert(),
            MetadataXpert(),
            IndexXpert(),
        ]

    drogas = HistoryIndexMoE()
    state = drogas.invoke({
        'input': 'Explain Mixture of Experts?'
    })

    print(state)



if __name__ == "__main__":
    main()
