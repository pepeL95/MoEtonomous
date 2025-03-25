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

class KeyRotatorEmbedder:
    """
    A wrapper around GoogleGenerativeAIEmbeddings that rotates among a list of API keys (licenses).
    This class attempts to generate embeddings using the current API key and, if an exception occurs (for example, due to rate limits or license issues), rotates to the next API key and retries.
    """
    def __init__(self, api_keys: list, embedding_model):
        if not api_keys:
            raise ValueError("At least one API key must be provided.")
        self.api_keys = api_keys
        self.embedding_model = embedding_model
        self.index = 0

    def rotate_key(self):
        """Rotates to the next API key in the list and updates the embedder."""
        self.index = (self.index + 1) % len(self.api_keys)

    def embed_documents(self, texts: list) -> list:
        """
        Generates embeddings for a list of documents using the current API key.
        If an exception occurs (e.g., due to license issues), rotates the key and retries once.
        """
        try:
            return self.embedding_model.embed_documents(texts)
        except Exception as e:
            # Optionally, log the exception here
            self.rotate_key()
            return self.embedding_model.embed_documents(texts)