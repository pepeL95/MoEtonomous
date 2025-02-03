from typing import Any

from moe.base.mixture import BaseMoE
from moe.base.strategies import BaseExpertStrategy

from agents.parsers.generic import StringParser
from moe.prebuilt.ragentive.postrieval.experts.factory import PostrievalDirectory


class RouterStrategy(BaseExpertStrategy):
    def execute(self, expert, state) -> dict[str, Any]:
        output = expert.invoke(state)
        return {'expert_output': output}


class RerankingStrategy(BaseExpertStrategy):
    def execute(self, expert, state) -> dict[str, Any]:
        if not 'context' in state['kwargs']:
            raise ValueError('Context documents not provided to the reranker')

        # Top k documents per query, where k is defined at expert init time
        outputs = set()
        for q_, docs in zip(state['kwargs']['enhanced_queries'], state['kwargs']['context']):
            local_outputs = expert.invoke({
                'input': q_,
                'context': docs,
            })
            outputs.update(local_outputs)

        state['kwargs']['context'] = StringParser.from_langdocs(list(outputs))
        state['expert_output'] = state['kwargs']['context']
        state['next'] = PostrievalDirectory.ContextExpert
        return state


class ContextStrategy(BaseExpertStrategy):
    def execute(self, expert, state) -> dict[str, Any]:
        output = expert.invoke({
            'input': state['input'],
            'topic': state['kwargs'].get('topic', 'No specific topic, go ahead and imply it as best as you can from the query'),
            'context': state['kwargs']['context']
        })

        state['expert_output'] = output
        state['next'] = BaseMoE.FINISH
        return state
