from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"]) # .env file path
    sys.path.append(os.environ.get('SRC'))

from MoE.config.debug import Debug
from dev_tools.utils.clifont import input_bold, print_cli_message

from MoE.mixtures.genChatMoE.genChatMoE import GenChatMoE
from MoE.xperts.expert_factory import ExpertFactory
from dev_tools.enums.llms import LLMs

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory


class GenChatMoERunner:
    @staticmethod
    def get():
        router = ExpertFactory.get(xpert=ExpertFactory.Directory.Router, llm=LLMs.GEMINI())
        gen_xpert = ExpertFactory.get(xpert=ExpertFactory.Directory.GeneralKnowledgeExpert, llm=LLMs.GEMINI())
        web_xpert = ExpertFactory.get(xpert=ExpertFactory.Directory.WebSearchExpert, llm=LLMs.GEMINI())

        gen_chat_MoE = GenChatMoE(
            name='GenChatOrchestrator',
            description=None,
            router=router,
            experts=[gen_xpert, web_xpert],
            verbose=Debug.Verbosity.low,
        ).build_MoE()

        return gen_chat_MoE


################################## REGRESSION TEST #########################################


if __name__ == '__main__':
    # Run
    memory = ChatMessageHistory()
    chat = GenChatMoERunner.get()
    print('*' * 100, '\n')
    while (user_input := input_bold('user: ')) != 'exit':
        state = chat.invoke({
            'input': user_input,
            'ephemeral_mem': memory,
            # 'expert_input': user_input,
            # 'next': ExpertFactory.Directory.GeneralKnowledgeExpert
        })
        memory.add_message(HumanMessage(content=user_input, role='user'))
        memory.add_message(AIMessage(content=state['expert_output'], role='assistant'))

        print_cli_message('**assistant: **' + state['expert_output'])