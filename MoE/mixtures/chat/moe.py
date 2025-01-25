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
from dev_tools.enums.llms import LLMs
from MoE.base.mixture.base_mixture import MoEBuilder
from dev_tools.enums.prompt_parsers import PromptParsers
from MoE.mixtures.chat.experts.factory import ChatFactory, ChatDirectory

if __name__ == '__main__':
    # Init Chat MoE
    chat = MoEBuilder()\
        .set_name('ChatMoE')\
        .set_description(None)\
        .set_router(ChatFactory.get(expert_name=ChatDirectory.Router, llm=None))\
        .set_verbosity(Debug.Verbosity.quiet)\
        .set_experts([
            ChatFactory.get(expert_name=ChatDirectory.GenXpert, llm=LLMs.Gemini(), prompt_parser=PromptParsers.Identity()),
            ChatFactory.get(expert_name=ChatDirectory.WebSearchXpert, llm=LLMs.Gemini(), prompt_parser=PromptParsers.Identity()),
        ])\
        .build()

    # Run loop
    while (user_input := input('user: ')) != 'exit':
        state = chat.invoke({
            'input': user_input,
        })

        print(state['expert_output'], end='\n\n')
