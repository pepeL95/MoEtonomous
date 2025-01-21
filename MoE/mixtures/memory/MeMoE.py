from MoE.base.expert import Expert
from MoE.base.mixture.base_mixture import MoE


class MeMoE(MoE):
    '''
    Non-trivial memory management

    Input: user query

    1. Topic xtraction from query + short-term memory (i.e. keep track of what we are talking about)
    2. Topic fitting for clustering + modifying synthesis of cluster (if topic deviates over threshold in semantic similarity from all clusters, add it) 
    3. Add metadata:
        - topic

    4. RAG

    Output: compressed context information + sliding window short-term memory
    '''


'''
# Observations:

* In a conversation history chronological timeline is important
* Topic modeling might be a helpful piece of information
* RAGetea to' lo vivo

'''