import torch
from enum import Enum
from typing import Tuple
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
from transformers.models.mpnet.modeling_mpnet import MPNetModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from transformers.models.mpnet.tokenization_mpnet_fast import MPNetTokenizerFast

from dev_tools.patterns.singleton import Singleton


class GoogleText004(Singleton):
    def __init__(self):
        self.ef = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

class LocalMPNetBaseV2(Singleton):
    def __init__(self):
        '''
        Returns langchain-huggingface embedding model.
        Note: The result from the embedding function will be a List[Float]
        '''

        device = 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'

        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        embedding_model_kwargs = {'device': device}
        # vector normalization so that dot product == cosine similarity
        encode_kwargs = {'normalize_embeddings': True}
        self.ef = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=embedding_model_kwargs,
            encode_kwargs=encode_kwargs
        )

class LocalMPNetBaseV2Tensor(Singleton):
    def __init__(self):
        '''
        Note: The result from the embedding function will be a torch.Tensor
        '''
        # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.ef = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")


class Embeddings(Enum):
    LocalMPNetBaseV2 = LocalMPNetBaseV2
    LocalMPNetBaseV2Tensor = LocalMPNetBaseV2Tensor
    GoogleText004 = GoogleText004

    def __call__(self):
        return self.value().ef