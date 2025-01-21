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

from dev_tools.utils.clifont import input_bold, print_bold

from MoE.base.mixture.base_mixture import MoEBuilder
from MoE.mixtures.chat.experts.factory import ChatFactory, ChatDirectory

if __name__ == '__main__':
    chat = MoEBuilder()\
        .set_name('ChatMoE')\
        .set_description(None)\
        .set_router(ChatFactory.get(expert_name=ChatDirectory.Router))\
        .set_verbosity(Debug.Verbosity.quiet)\
        .set_experts([
            ChatFactory.get(expert_name=ChatDirectory.GenXpert), 
            ChatFactory.get(expert_name=ChatDirectory.WebSearchXpert),
            ])\
        .build()
    
    while (user_input := input_bold('user: ')) != 'exit':
        state = chat.invoke({
            'input': user_input,
        })
        
        print_bold(state['expert_output'])