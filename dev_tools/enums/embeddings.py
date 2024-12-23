import torch
from typing import Tuple
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
from transformers.models.mpnet.modeling_mpnet import MPNetModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from transformers.models.mpnet.tokenization_mpnet_fast import MPNetTokenizerFast

class Embeddings:
    @staticmethod
    def google_text_embedding_004() -> Embeddings:
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    @staticmethod
    def sentence_transformers_mpnet() -> HuggingFaceEmbeddings:
        '''
        Returns langchain-huggingface embedding model.
        
        Note: Observe we do not return a tokenizer in this class
        Note: The result from the embedding function will be a List[Float]
        '''

        device = 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
        
        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        embedding_model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True} # vector normalization so that dot product == cosine similarity
        ef = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=embedding_model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        return ef
    
    @staticmethod
    def auto_model_mpnet() -> Tuple[MPNetModel, MPNetTokenizerFast]:
        '''
        Returns hf embedding model and tokenizer.
        
        Note: The result from the embedding function will be a torch.Tensor
        '''
        # Load tokenizer + encoder
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        embedding_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        return tokenizer, embedding_model

        


    