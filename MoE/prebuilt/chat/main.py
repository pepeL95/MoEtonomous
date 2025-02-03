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
from moe.base.mixture import MoEBuilder
from moe.prebuilt.chat.experts.factory import ChatFactory, ChatDirectory

if __name__ == '__main__':
    # Init Chat MoE
    chat = MoEBuilder()\
        .set_name('ChatMoE')\
        .set_description(None)\
        .set_router(ChatFactory.get(expert_name=ChatDirectory.Router))\
        .set_verbosity(Debug.Verbosity.low)\
        .set_experts([
            ChatFactory.get(expert_name=ChatDirectory.GenXpert),
            ChatFactory.get(expert_name=ChatDirectory.WebSearchXpert),
        ])\
        .build()

    # Run loop
    while (user_input := input('user: ')) != 'exit':
        state = chat.invoke({
            'input': user_input,
        })

        print(state['expert_output'], end='\n\n')
