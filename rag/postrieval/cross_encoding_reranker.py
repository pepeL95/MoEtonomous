from dev_tools.utils.clifont import CLIFont, print_bold

from langchain_core.runnables import RunnableLambda
from langchain_core.documents.base import Document
from langchain_core.runnables import Runnable

import numpy as np
import functools

class CrossEncodingReranker:
    def __init__(self, cross_encoder: Runnable):
        self.cross_encoder = cross_encoder

    def _computeCrossEncodingReRanking(self, input: dict, rerank_kwargs: dict, verbose: bool = False):
        # Check for required keys
        rerank_kwargs.setdefault('k', 5)
        rerank_kwargs.setdefault('score_heuristic', lambda score: score)

        query = input.get('input')
        docs = input.get('context')

        assert query and docs

        # Make sure we get a string as the output
        if docs and isinstance(docs[0], Document):
            docs = [doc.page_content for doc in docs]

        # Build the pairs to apply sentence re-ranking using cross-encoders
        pairs = [(query, doc) for doc in docs]

        # Apply cross-encoding
        scores = self.cross_encoder.invoke(pairs)

        # Apply heuristic
        scores = [rerank_kwargs.get('score_heuristic')(score) for score in scores]

        # Sort scores
        ranked_scores_i = np.argsort(scores)[::-1]

        # Context is now the most relevant documents to the original user query
        relevant_documents = [
            # Return the top k
            docs[i] for i in ranked_scores_i[:rerank_kwargs['k']]
        ]

        # Display results
        if verbose:
            for document in relevant_documents:
                print_bold(f'\n{CLIFont.light_green}{document}')
                print('*' * 100)
        return relevant_documents

    def as_reranker(self, rerank_kwargs: dict = {'k': 5, 'score_heuristic': lambda score: score}, verbose=False) -> RunnableLambda:
        # Type check for kwargs
        if not isinstance(rerank_kwargs, dict):
            raise ValueError(f"rerank_kwargs must be a dict. Got {type(rerank_kwargs)}.")
       
        _reranker = functools.partial(self._computeCrossEncodingReRanking, rerank_kwargs=rerank_kwargs, verbose=verbose)
       
        return RunnableLambda(_reranker)
