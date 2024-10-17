from sentence_transformers import CrossEncoder

class CrossEncodings:
    '''
    Static class for standardizing cross encoding models
    '''
    
    @staticmethod
    def sentence_transformer_miniLM():
        cross_encoding_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return cross_encoding_model