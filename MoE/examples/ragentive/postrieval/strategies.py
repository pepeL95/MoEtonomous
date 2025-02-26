from typing import Any

from moe.base.mixture import BaseMoE
from moe.base.strategies import BaseExpertStrategy

from agents.parsers.generic import StringParser


class RerankingStrategy(BaseExpertStrategy):
    def execute(self, expert, state:BaseMoE.State) -> dict[str, Any]:
        if not 'contexts' in state['kwargs']:
            raise ValueError('Context documents not provided to the reranker')

        outputs = set()
        # For each enhanced query, rerank docs w.r.t enhanced queries
        for i, query in enumerate(state['kwargs']['enhanced_queries']):
            local_outputs = expert.invoke({
                'input': query,
                'context': state['kwargs']['contexts'][i],
            })

            outputs.update(local_outputs)


        state['kwargs']['context'] = StringParser.from_langdocs(list(outputs))
        state['expert_output'] = state['kwargs']['context']
        state['next'] = 'ContextExpert'
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
