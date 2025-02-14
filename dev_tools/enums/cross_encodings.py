from enum import Enum
from sentence_transformers import CrossEncoder
from dev_tools.patterns.singleton import Singleton
from langchain_core.runnables import RunnableLambda

class LocalMiniLM(Singleton):
    def __init__(self):
        print('initializing...')
        cross_encoding_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.model = RunnableLambda(lambda input : cross_encoding_model.predict(input))


class CrossEncodings(Enum):
    LocalMiniLM = LocalMiniLM

    def __call__(self):
        return self.value().model
    
__all__ = ["CrossEncodings"]