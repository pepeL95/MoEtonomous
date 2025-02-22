from moe.base.mixture import BaseMoE
from moe.base.strategies import BaseExpertStrategy
    
class PretrievalStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['input'],
            'kwargs': {
                'topic': state['kwargs']['topic'],
            }
        })
        state['expert_output'] = "Successfully finished the pre-retrieval step of the RAG pipeline."
        state['kwargs']['hyde'] = output['kwargs']['hyde']
        state['kwargs']['enhanced_queries'] = output['kwargs']['enhanced_queries']
        state['next'] = 'Retrieval'
        return state

class RetrievalStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        # 2 dim array with the top 10 documents per search query
        outputs = []
        for _q in state['kwargs']['hyde']:
            relevant_docs = expert.invoke(_q)
            outputs.append(relevant_docs)

        state['expert_output'] = 'Successfully retrieved relevant documents.'
        state['kwargs']['context'] = outputs
        state['next'] = 'Postrieval'
        return state

class PostrievalStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['input'],
            'kwargs': state['kwargs'],
        })
        state['expert_output'] = output['expert_output']
        state['next'] = BaseMoE.FINISH
        return state
