from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"]) # .env file path
    sys.path.append(os.environ.get('SRC'))

from MoE.base.mixture import MoE
from MoE.config.debug import Debug
from MoE.xperts.expert_factory import ExpertFactory
from MoE.mixtures.listpal.listpalMoE import ListPalMoE

from dev_tools.enums.llms import LLMs
from dev_tools.utils.clifont import input_bold, print_cli_message

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

class ListPalMoERunner:
    def get() -> MoE:
        # Create router + experts
        router = ExpertFactory.get(xpert=ExpertFactory.Directory.Router, llm=LLMs.GEMINI())
        jira_xpert = ExpertFactory.get(xpert=ExpertFactory.Directory.JiraExpert, llm=LLMs.GEMINI())
        jira_xpert.description = "Excellent task manager assistant. Has an expertise in managing lists, and other productivity-related tasks. Make sure to remind the expert that the project id is `LIST` when in doubt."
        jira_xpert.agent.system_prompt = (
            "You are an exceptional task management assistant. Your domain is specialized in managing jira issues.\n"
            "Your task is to fulfill the user's request by managing the user's Jira project with project id: `LIST`.\n"
            "You have some tools at your disposal, use them wisely.\n"
            "Provide a **highly detailed** report of your actions ands results, increasing transparency.\n"
            "Use an assitant-like tone in your responses."
        )


        # Init MoE
        listpalMoE = ListPalMoE(
            name='ListPalMoE',
            description=None,
            router=router,
            experts=[jira_xpert],
            verbose=Debug.Verbosity.low
        ).build_MoE()

        return listpalMoE

################################## REGRESSION TEST #########################################


if __name__ == '__main__':
    # Memory
    mem = ChatMessageHistory()

    # Run
    listpal = ListPalMoERunner.get()
    while (user_input := input_bold('user: ')) != 'exit':
            # Call MoE
            state = listpal.invoke({
                'input': user_input,
                'ephemeral_mem': mem
                # 'next': ExpertFactory.Directory.JiraExpert,
                # 'expert_input': user_input,
            })
            
            # Update chat history
            mem.add_message(HumanMessage(content=user_input, role='user'))
            mem.add_message(AIMessage(content=state['expert_output'], role='assistant'))
            
            # Print to console 
            print_cli_message('**assistant: **' + state['expert_output'])

    
    
