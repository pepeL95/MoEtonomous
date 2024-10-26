from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"]) # .env file path
    sys.path.append(os.environ.get('SRC'))


from JB007.base.ephemeral_nlp_agent import EphemeralNLPAgent
from JB007.base.ephemeral_tool_agent import EphemeralToolAgent
from JB007.prompters.prompters import Prompters
from JB007.parsers.pydantic import ArxivPyParser
from JB007.toolbox.toolbox import Toolbox

from MoE.base.expert import Expert
from MoE.base.router import Router
from MoE.config.debug import Debug
from MoE.mixtures.arxiv.arxivMoE import ArxivMoE

from dev_tools.enums.llms import LLMs

from langchain_core.runnables import RunnableLambda

from dev_tools.utils.clifont import print_cli_message


class ArxivMoERunner:
    @staticmethod
    def get():
        # Create router & xperts
        router = Router(
            name='ArxivOrchestrator',
            description=None,
            agent=RunnableLambda(lambda input: (
                f'\nAction: ArxivQbuilderXpert\n'
                f'Action Input: {input['input']}\n'
        )))

        query_xpert = Expert(
            name='ArxivQbuilderXpert',
            description='Dexterous at taking a search query and converting it into a valid JSON format for a downstream search task: searching the Arxiv api for scholar papers.',
            agent=EphemeralNLPAgent(
                name='ArxivQbuilderAgent',
                llm=LLMs.GEMINI(),
                system_prompt=(
                    'You are an dexterous at taking a search query and converting it into a valid format for searching the Arxiv api for scholar papers. '
                    'Consider the user query and follow the instructions thoroughly'
                    ),
                prompt_template=Prompters.Arxiv.ApiQueryBuildFewShot(),
                parser=ArxivPyParser.apiQueryJson()
            )
        )

        search_xpert = Expert(
            name='ArxivSearchXpert',
            description='An Arxiv api search expert. It excels at the following task: given a valid JSON query, it executes the query, searching and fetching papers from the Arxiv system.',
            agent=EphemeralToolAgent(
                name='ArxivSearchAgent',
                llm=LLMs.GEMINI(),
                system_prompt=(
                    'You are a search expert, specialized in searching the Arxiv api for scholar papers.\n'
                    'Your task is to build a query and then execute it.\n' 
                    'You have some tools at your disposal, use them wisely, in the right order.'
                ),
                tools=[Toolbox.Arxiv.build_query, Toolbox.Arxiv.execute_query],
            )
        )

        sigma_xpert = Expert(
            name='ArxivSigmaXpert',
            description=(
                'An NLP Guru. It specializes in summarization and feature extraction tasks. '
                'Useful expert when we need to synthesize information and provide insights from obtained results.'
            ),
            agent=EphemeralNLPAgent(
                name='ArxivSigmaAgent',
                system_prompt='You are an nlp expert, specialized in feature extraction and summarization.',
                prompt_template=Prompters.Arxiv.AbstractSigma(),
                llm=LLMs.GEMINI(),
            )
        )

        # Init & return MoE
        arxivMoE = ArxivMoE(
            name='ArxivMoE',
            description=None,
            router=router,
            experts=[query_xpert, search_xpert, sigma_xpert],
            verbose=Debug.Verbosity.low,
        ).build_MoE()

        return arxivMoE


################################## REGRESSION TEST #########################################


if __name__ == '__main__':
    # Run
    arxiv = ArxivMoERunner.get()
    user_input = input('Enter query: ')
    state = arxiv.invoke({'input': user_input})
    print_cli_message('**assistant: **' + state['expert_output'])