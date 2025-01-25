from JB007.base.ephemeral_nlp_agent import EphemeralNLPAgent
from MoE.base.expert.base_expert import Expert

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser


class Router(Expert):
    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None, prompt_parser=None):
        super().__init__(
            description=description or Router.__doc__,
            name=name or Router.__name__,
            strategy=strategy,
            agent=agent or RunnableLambda(lambda state: (
                f"\nAction: {QueryAugmentationExpert.__name__}"
                f"\nAction Input: {state['input']}"
            )),
        )


class QueryAugmentationExpert(Expert):
    '''A search query extraction master. It extracts and decouples queries to optimize information retrieval tasks.'''

    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None, prompt_parser=None):
        if llm is None:
            raise ValueError('LLM cannot be None')

        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or QueryAugmentationExpert.__name__,
            description=description or QueryAugmentationExpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
                llm=llm,
                prompt_parser=prompt_parser,
                name='QueryAugmentationAgent',
                system_prompt=(
                    "You are an expert at optimizing queries, extracting implicit search structures from complex user aiming for efficient information retrieval. "
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
                    "You: {{\n"
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
                output_parser=JsonOutputParser(),
            )
        )


class HydeExpert(Expert):
    '''Master at generating hypothetical documents to provide better similarity search results in a Retrieval Augmented Generation (RAG) and Information Retrieval (IR) pipeline'''

    def __init__(self, agent=None, description=None, name=None, strategy=None, llm=None, prompt_parser=None):
        if llm is None:
            raise ValueError('LLM cannot be None')

        if strategy is None:
            raise ValueError('strategy cannot be None')

        super().__init__(
            name=name or HydeExpert.__name__,
            description=description or HydeExpert.__doc__,
            strategy=strategy,
            agent=agent or EphemeralNLPAgent(
                llm=llm,
                prompt_parser=prompt_parser,
                name='HydeAgent',
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
        )
