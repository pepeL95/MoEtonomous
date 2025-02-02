from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableLambda

class CrossEncodings:
    @staticmethod
    def sentence_transformer_miniLM():
        cross_encoding_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        ret = RunnableLambda(lambda input : cross_encoding_model.predict(input))
        return ret
