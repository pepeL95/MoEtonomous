from moe.base.strategies import BaseExpertStrategy
from moe.base.mixture import BaseMoE
from langchain_core.documents import Document

class ThoughtStrategy(BaseExpertStrategy):
    def execute(self, expert, state:BaseMoE.State):
        reasoning = expert.invoke({
            'input': state['expert_input']
        })
    
        state['expert_output'] = "Reasoned about query. See results in state['kwargs']['reasoning']"
        state['kwargs']['reasoning'] = reasoning
        state['next'] = 'ResponseXpert'

        return state
    
class ResponseStrategy(BaseExpertStrategy):
    def execute(self, expert, state:BaseMoE.State):
        expert_input = {
            'query': state['expert_input'],
            'reasoning': state['kwargs']['reasoning']
        }
        response = expert.invoke({
            'input': expert_input
        })
    
        state['expert_output'] = "Response generated. See response in state['kwargs']['response']"
        state['kwargs']['response'] = response
        state['next'] = 'MetadataXpert'

        return state
    
class MetadataXStrategy(BaseExpertStrategy):
    def execute(self, expert, state:BaseMoE.State):
        expert_input = {
            'query': state['expert_input'],
            'reasoning': state['kwargs']['reasoning'],
            'response': state['kwargs']['response']
        }
        metadata = expert.invoke({
            'input': expert_input
        })
    
        state['expert_output'] = "Metadata generated. See metadata in state['kwargs']['metadata']"
        state['kwargs']['metadata'] = metadata
        state['next'] = 'IndexXpert'

        return state

class IndexingStrategy(BaseExpertStrategy):
    def execute(self, expert, state:BaseMoE.State):
        query = state['expert_input']
        reasoning = state['kwargs']['reasoning']
        response = state['kwargs']['response']
        metadata = state['kwargs']['metadata']
        response_page_content = f'''**Query**: {query} \n **Reasoning**: {reasoning} \n **Response**: {response}'''
        response_doc = Document(
            page_content=response_page_content,
            metadata=metadata
        )
        expert.agent.load_document(response_doc, 'moeses_response_index')

        metadata_page_content = f'''**Query**: {query} \n **Reasoning**: {reasoning}'''
        meta_doc = Document(
            page_content=metadata_page_content,
            metadata=metadata
        )
        expert.agent.load_document(meta_doc, 'moeses_meta_index')
    
        state['expert_output'] = 'Indexing complete'
        state['next'] = BaseMoE.FINISH

        return state
    