from dotenv import load_dotenv
import sys
import os

if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"]) # .env file path
    sys.path.append(os.environ.get('SRC'))

from MoE.mixtures.raggaeton.postrieval_MoE import PostrievalMoE
from MoE.xperts.expert_factory import ExpertFactory
from MoE.config.debug import Debug
from MoE.base.router import Router

from dev_tools.enums.cross_encodings import CrossEncodings
from dev_tools.utils.clifont import print_cli_message
from dev_tools.enums.llms import LLMs

from langchain_core.runnables import RunnableLambda

from RAG.base.cross_encoding_reranker import CrossEncodingReranker

class PostRetrievalMoE:
    @staticmethod
    def get():

        # Init experts
        router_ = Router(
            name='PostRetrievalOrchestrator',
            description='Orchestrates the RAG experts in the post-retrieval step in a modular RAG pipeline',
            agent=RunnableLambda(lambda input : (
                    f"\nAction: {ExpertFactory.Directory.RerankingExpert}\n"
                    f"Action Input: {input['input']}\n"
                )
            )
        )
        
        reranker_xpert = ExpertFactory.get(
            xpert=ExpertFactory.Directory.RerankingExpert, 
            llm=None, 
            reranker=CrossEncodingReranker(
                cross_encoder=CrossEncodings.sentence_transformer_miniLM()
            ).as_reranker(rerank_kwargs={'k': 8})
        )
        
        context_xpert = ExpertFactory.get(
            xpert=ExpertFactory.Directory.ContextExpert, 
            llm=LLMs.Gemini(),
        )

        # Init MoE
        postrievalMoE = PostrievalMoE(
            name='PostrievalMoE',
            description=(
                    'Expert at coordinating the post-retrieval step of a Retrieval Augmented Generation (RAG) pipeline. '
                    'Use this expert when the pre-retrival step is done.'
                    'You may END after this expert has responded.'
                ),
                router=router_,
                experts=[reranker_xpert, context_xpert],
                verbose=Debug.Verbosity.low,
        ).build_MoE()

        return postrievalMoE
        

################################## REGRESSION TEST #########################################

if __name__ == '__main__':
    # Run
    post_retrievalMoE = PostRetrievalMoE.get()
    
    user_input = input('Enter prompt: ')

    hyde = [
        "AWS Lambda's pay-as-you-go pricing model offers several benefits, including:\n\n* **Cost-effectiveness:** You only pay for the compute time your functions consume, making it ideal for applications with fluctuating workloads.\n* **Scalability:** Lambda automatically scales your functions based on demand, ensuring your application can handle spikes in traffic without requiring manual intervention.\n* **Reduced operational overhead:** You don't need to manage servers or infrastructure, allowing you to focus on developing your applications.\n* **Faster time to market:** Lambda's serverless nature allows you to deploy and iterate on your applications quickly. \n"
    ]
    
    search_queries = [
        "AWS Lambda pay as you go pricing model"
    ]

    state = post_retrievalMoE.invoke({'input': user_input, 'kwargs': {'topic': 'AWS Lambda', 'hyde': hyde, 'search_queries': search_queries}})
    print_cli_message('**assistant: **' + state['expert_output'])
    