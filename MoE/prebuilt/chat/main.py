############################# SYS PATH LOAD ################################

from dotenv import load_dotenv
import sys
import os

from dev_tools.enums.llms import LLMs
from moe.annotations.core import Autonomous, ForceFirst, MoE
from moe.default.strategies import DefaultMoEStrategy

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"])  # .env file path
    sys.path.append(os.environ.get('SRC'))

###########################################################################

from agents.config.debug import Debug
from moe.base.mixture import MoEBuilder
from moe.prebuilt.chat.experts import Factory

if __name__ == '__main__':    
    
    @MoE(DefaultMoEStrategy)
    @Autonomous(LLMs.Gemini())
    class ChatMoE:
        '''Chat MoE with General and Websearch functionality'''
        experts = [
            Factory.get(expert_name=Factory.Dir.GenXpert),
            Factory.get(expert_name=Factory.Dir.WebSearchXpert),
        ]
    
    
    # Run
    chat = ChatMoE()
    while (user_input := input('user: ')) != 'exit':
        state = chat.invoke({
            'input': user_input,
        })

        print(state['expert_output'], end='\n\n')
