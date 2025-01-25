from MoE.base.mixture.base_mixture import MoE
from MoE.mixtures.ragentive.modular.experts.factory import RagDirectory
from MoE.base.strategy.expert.base_strategy import BaseExpertStrategy


class RouterStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke(state)
        return {'expert_output': output}
    
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
        state['kwargs']['search_queries'] = output['kwargs']['search_queries']
        state['next'] = RagDirectory.Retrieval
        return state

class RetrievalStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        # n-d array with the top 10 documents per search query
        outputs = []
        for _q in state['kwargs']['hyde']:
            local_outputs = expert.invoke(_q)
            outputs.append(local_outputs)

        state['expert_output'] = 'Successfully retrieved relevant documents.'
        state['kwargs']['context'] = outputs
        state['next'] = RagDirectory.PostrievalMoE
        return state

class PostrievalStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['input'],
            'kwargs': state['kwargs']
        })
        state['expert_output'] = output['expert_output']
        state['next'] = MoE.FINISH
        return state
