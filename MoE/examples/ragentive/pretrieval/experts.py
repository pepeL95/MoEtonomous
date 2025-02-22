from langchain_core.output_parsers import JsonOutputParser

from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent

from moe.annotations.core import Expert
from moe.examples.ragentive.pretrieval.strategies import HydeStrategy, QueryAugmentationStrategy

from dev_tools.enums.llms import LLMs


@Expert(QueryAugmentationStrategy)
class QueryAugmentationExpert:
    '''A search query extraction master. It extracts and decouples queries to optimize information retrieval tasks.'''
    agent =  EphemeralNLPAgent(
        llm=LLMs.Gemini(),
        name='QueryAugmentationAgent',
        system_prompt=(
            "You are an expert at optimizing queries, extracting implicit search structures from complex ones aiming for efficient information retrieval. "
            "You thrive at engineering direct queries without adding any unnecessary information"
        ),
        prompt_template=(
            "### Instructions:\n"
            "1. Break down compound user queries into distinct single queries, aimed to be individual inputs to an intelligent search agent.\n"
            "2. Extract implicit search intentions within the user query.\n"
            "3. Rewrite each query to make it more precise, ensuring it targets the most relevant information.\n"
            "4. Use the topic provided (if any) for scoping the queries.\n"
            "5. Provide your reasoning as for why you extracted the given query.\n\n"
            "### Example\n"
            "Topic: Electric Vehicles\n"
            "User: Tell me about electric cars, focus on latest models and how they compare to hybrids in terms of fuel efficiency. Be brief\n"
            "You: {{\n"
            "  \"queries\": [\n"
            "    {{\n"
            "      \"query\": \"Latest electric vehicle models and their key specifications\",\n"
            "      \"reasoning\": \"The user is interested in the latest electric cars, so this query focuses on retrieving up-to-date EV models along with relevant details.\"\n"
            "    }},\n"
            "    {{\n"
            "      \"query\": \"Comparison of electric vehicles and hybrid cars in terms of fuel efficiency\",\n"
            "      \"reasoning\": \"The user explicitly requests a comparison between EVs and hybrids regarding fuel efficiency, making this a necessary distinct query.\"\n"
            "    }},\n"
            "  ]\n"
            "}}\n\n"
            "### Output format:\n"
            "Return a JSON in the following format:\n"
            "{{\n"
            "  \"queries\": [{{\"query\": <An enhanced query. Each query is a refined, standalone search input>, \"reasoning\": <A brief reasoning explaining why it was extracted based on the user's request.>}}, ...],\n"
            "}}\n\n"
            "### Query:\n"
            "Topic: {topic}\n"
            "User: {input}\n"
            "You: "
        ),
        output_parser=JsonOutputParser(),
    )


@Expert(HydeStrategy)
class HydeExpert:
    '''Master at generating hypothetical documents to provide better similarity search results in a Retrieval Augmented Generation (RAG) and Information Retrieval (IR) pipeline'''
    agent = EphemeralNLPAgent(
        llm=LLMs.Gemini(),
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


####################################################################################################

class Factory:
    class Dir:
        QueryAugmentationExpert: str =  'QueryAugmentationExpert'
        HydeExpert: str = 'HydeExpert'

    @staticmethod
    def get(expert_name:str, **kwargs):
        if expert_name == Factory.Dir.QueryAugmentationExpert:
            return QueryAugmentationExpert(**kwargs)
        if expert_name == Factory.Dir.HydeExpert:
            return HydeExpert(**kwargs)

        raise ValueError(f'No expert by name {expert_name} exists.')
