from typing import Union
from dev_tools.enums.llms import LLMs

from MoE.base.expert import Expert
from MoE.base.router import Router
from MoE.prompts.prompt_repo import PromptRepo

from RAG.base.cross_encoding_reranker import CrossEncodingReranker

from JB007.toolbox.toolbox import Toolbox
from JB007.base.ephemeral_nlp_agent import EphemeralNLPAgent
from JB007.base.persistent_nlp_agent import PersistentNLPAgent
from JB007.base.persistent_tool_agent import PersistentToolAgent

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser



class ExpertRepo:
    '''
    Repository for specialized agents (a.k.a. experts). Defaults to the GEMINI llm.
    '''
    class Router:
        '''You are a master at managing conversations between the user and multimple experts. You must autonimously decides where to route inputs/outputs to.'''
        @staticmethod
        def get_router(llm=LLMs):
            agent = PersistentNLPAgent(
                name=ExpertRepo.Router.__name__,
                llm=llm,
                prompt_template=PromptRepo.router_react(), # template variables: experts, expert_names, input, scratchpad.
                system_prompt=(
                "You are an assistant managing a conversation with the user.\n"
                "You can leverage multiple experts who collaborate to fulfill the user's query.\n"
                "Consider the conversation history."
                ),
            )
            router = Router(
                name=ExpertRepo.Router.__name__,
                description=ExpertRepo.Router.__doc__,
                agent=agent,
            )
            return router
        
    class GeneralKnowledgeExpert:
        '''Excellent expert on a wide range of topics such as coding, math, history, an much more!!. Default to this expert when not sure which expert to use.'''
        @staticmethod
        def get_expert(llm:LLMs) -> Expert:
            chat_agent = PersistentNLPAgent(
                name=ExpertRepo.GeneralKnowledgeExpert.__name__,
                llm=llm,
                system_prompt=(
                    "You are a general knowledge expert who thrives in giving accurate information.\n"
                    "You are part of a conversation with other experts who, together, collaborate to fulfill a user request.\n"
                    "Your input is given from another expert who needs you to answer it.\n"
                    "You are chosen for a reason! Do not ask for clarifications.\n"
                    "Respond to your queries with brief, fact-based answers as best as you can\n"
                    "Format your response nicely, using markdown."
                ),
            )
            chat_expert = Expert(
                agent=chat_agent,
                description=ExpertRepo.GeneralKnowledgeExpert.__doc__
            )
            return chat_expert
    
    class WebSearchExpert:
        '''Excels at searching the web for gathering up-to-date and real-time information. '''
        # '''Excellent at handling assistant-like tasks (e.g. web searching, scheduling meetings, accessing information about current events, and real-time requests)'''

        @staticmethod
        def get_expert(llm:LLMs) -> Expert:
            tool_agent = PersistentToolAgent(
                name=ExpertRepo.WebSearchExpert.__name__,
                llm=llm,
                tools=[Toolbox.Websearch.duck_duck_go_tool()],
                parser=StrOutputParser(),
                system_prompt=(
                    "You must always use the duck_duck_go_tool provided before you respond, **always!!**.\n\n"
                    "You are an online search and information gatherer expert.\n"
                    "You are part of a conversation with other experts who, together, collaborate to fulfill a request.\n"
                    "Your input is given from another expert who needs you to answer it.\n"
                    "You are chosen for a reason! Do not ask for clarifications.\n"
                    "Provide a **highly detailed** summary of the results you obtain, including sources.\n"
                    "Format your response nicely, using markdown."
                ),
            )
            realtime_expert = Expert(
                agent=tool_agent, 
                description=ExpertRepo.WebSearchExpert.__doc__
            )
            return realtime_expert
    
    class JiraExpert:
        '''Excellent at managing all Jira-related actions. Delegate all Jira-related tasks to this expert, it has all the information it needs already.'''
        
        @staticmethod
        def get_expert(llm:LLMs) -> Expert:
            jira_expert = PersistentToolAgent(
                llm=llm,
                name=ExpertRepo.JiraExpert.__name__,
                system_prompt=(
                    "You are a project management guru, specialized in managing jira issues and projects.\n"
                    "Your task is to fulfill the user's request by accessing Jira information.\n"
                    "You have some tools at your disposal, use them wisely.\n"
                    "Provide a **highly detailed** report of your actions ands results, increasing transparency."
                    ),
                tools=[
                    Toolbox.Jira.jql_query_tool,
                    Toolbox.Jira.create_jira_issue,
                    Toolbox.Jira.update_jira_issue,
                    Toolbox.Jira.transition_issue_state,
                    ],
                parser=StrOutputParser(),
                # verbose=True,
            )
            realtime_expert = Expert(
                agent=jira_expert, 
                description=ExpertRepo.JiraExpert.__doc__
            )
            return realtime_expert
    
    class QueryXtractionXpert:
        '''A search query extraction master. It extracts and decouples queries to optimize information retrieval tasks.'''

        def get_expert(llm:LLMs) -> Expert:
            quary_augmentation_agent = EphemeralNLPAgent(
                llm=llm,
                name=ExpertRepo.QueryXtractionXpert.__name__,
                system_prompt=(
                    "You are an expert query optimizer specializing in extracting implicit search queries from complex user inputs and rewriting them for precise and efficient information retrieval. "
                    "You thrive at engineering direct queries without adding any unnecessary information"
                    ),
                prompt_template=(
                    "### Instructions:\n"
                    "1. Break down compound user queries into distinct single queries, aimed to be individual inputs to an intelligent search agent.\n"
                    "2. Extract implicit search intentions within the user query.\n"
                    "3. Rewrite each query to make it more precise, ensuring it targets the most relevant information.\n"
                    "4. Use the topic provided (if any) for scoping the queries.\n"
                    "5. Provide your reasoning as for why you extracted the given query.\n\n"
                    " ### Example\n"
                    "Topic: Electric Vehicles\n"
                    "User: Tell me about electric cars, especially the latest models and how they compare to hybrids in terms of fuel efficiency. Be brief\n"
                    "{{\n"
                    "   \"search_queries\": [\"Give me an overview of electric cars.\", \"What are some of the latest models of electric cars?\", \"Draw a comparison between electric cars and hybrid cars on fuel efficiency.\"],\n"
                    "   \"reason\": \"There were three implicit queries. Here is what I did to build the queries:\n    1. Identify the first implicit search:  'Give me an overview of electric cars.'\n    2. Identify the second implicit search: 'What are some of the latest models of electric cars?'\n    3. Identify the third implicit search: 'Draw a comparison between electric cars and hybrid cars on fuel efficiency.'\n"
                    "}}\n\n"
                    "### Output format:\n"
                    "Return a JSON in the following format:\n"
                    "{{\n"
                    "   \"search_queries\": [\"<query_1>\", ...],\n"
                    "   \"reason\": \"<your reasoning for decoupling the search queries>\"\n"
                    "}}\n\n"
                    "### Query:\n"
                    "Topic: {topic}\n"
                    "User: {input}\n"
                    "You: "
                ),
                parser=JsonOutputParser()
            )

            query_augmentation_xpert = Expert(
                agent=quary_augmentation_agent,
                description=ExpertRepo.QueryXtractionXpert.__doc__
            )

            return query_augmentation_xpert
        
    class HyDExpert:
        '''Master at generating hypothetical documents to provide better similarity search results in a Retrieval Augmented Generation (RAG) and Information Retrieval (IR) pipeline'''

        @staticmethod
        def get_expert(llm:LLMs) -> Expert:
            hyDEAgent = EphemeralNLPAgent(
                llm=llm,
                name=ExpertRepo.HyDExpert.__name__,
                system_prompt=(
                    "You thrive in answering every query that is given tou you, always! "
                    "Your response will be used in a downstream information retrieval pipeline. "
                    "You have been chosen for a reason. Do not ask for clarifications!!"
                ),
                prompt_template=(
                    "## Instructions\n"
                    "- Write a small passage to answer the user query.\n"
                    "- If a context is provided, use it. Otherwise, use your knowledge base.\n"
                    "- If you do not know the answer you must still answer it with a hypothetical answer. But, scope it to the topic provided.\n"
                    "\n"
                    "### Topic\n"
                    "{topic}\n"
                    "\n"
                    "### Context:\n"
                    "{context}\n"
                    "\n"
                    "### User Query:\n"
                    "{input}\n"
                    "\n"
                    "### Passage:\n"
                )
            )
            
            hyDExpert = Expert(
                agent=hyDEAgent, 
                description=ExpertRepo.HyDExpert.__doc__
            )
            
            return hyDExpert
    
    class RetrieverExpert:
        '''Expert at retrieving semantically relevant documents with respect to a given query'''

        def get_expert(retriever:VectorStoreRetriever):
            retriever_expert = Expert(
                name=ExpertRepo.RetrieverExpert.__name__,
                agent=retriever,
                description=ExpertRepo.RetrieverExpert.__doc__
            )

            return retriever_expert

    class RerankingExpert:
        '''A master at ranking the relevance of retrieved documents with respect to a given query. It usually does its work after the RetrieverExpert'''

        def get_expert(reranker:Union[LLMs, CrossEncodingReranker]):
            reranker_expert = Expert(
                name=ExpertRepo.RerankingExpert.__name__,
                agent=reranker,
                description=ExpertRepo.RetrieverExpert.__doc__
            )

            return reranker_expert
        
    class ContextExpert:
        '''Master at giving informed answers. It uses a given context to augment its knowledge.'''

        def get_expert(llm:LLMs) -> Expert:
            context_agent = EphemeralNLPAgent(
                llm=llm,
                name=ExpertRepo.ContextExpert.__name__,
                system_prompt="You are an expert at giving informed answers.",
                prompt_template=(
                    "Use the following topic and pieces of retrieved context to enhance your knowledge\n"
                    "Answer the user query as best as possible\n"
                    "If you don't know the answer, try your best to answer anyways. \n"
                    "Be comprehensive with your answers.\n\n"
                    "### Topic\n"
                    "{topic}. \n\n"
                    "### Query:\n"
                    "{input}\n\n"
                    "### Context:\n"
                    "{context}\n\n"
                    "### Answer:\n"
                ),
            )
            
            context_expert = Expert(
                agent=context_agent,
                description=ExpertRepo.ContextExpert.__doc__
            )

            return context_expert