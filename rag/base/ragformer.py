from JB007.parsers.generic import StringParser

import math
import torch
import functools
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple
from langchain_core.documents.base import Document
from langchain_core.runnables import RunnableLambda, Runnable
from transformers.models.mpnet.modeling_mpnet import MPNetModel
from transformers.models.mpnet.tokenization_mpnet_fast import MPNetTokenizerFast

class RagFormer:
    def __init__(
            self, 
            tokenizer: MPNetTokenizerFast, 
            embedding_model: MPNetModel,
            X: torch.Tensor=torch.zero_,
            docs: List[str] = None,
            device: Optional[str] = None,
            ) -> None:
        
        # Set up gpu if available
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'    
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        # Init params
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model.to(self.device)
        self.X = X
        self.docs = docs

        # Get embeddings if none provided
        if self.X == torch.zero_:
            self.X , _ = self.get_embeddings(self.docs)
            
    def get_embeddings(self, texts: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Computes the embedding vectors for the given texts. It returns a tuple
        specifying: tuple(embeddings, attentions)

        Parameters:
        - texts: a string or list of strings to compute the embeddings.
        '''
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs, output_attentions=True)
        return outputs.last_hidden_state, outputs.attentions

    def retrieve_with_attention(self, q: torch.Tensor, X: torch.Tensor) -> List[torch.Tensor]:
        '''
        Performs scaled dot product attention with Q and K as the query and document 
        embeddings respectively. 
        
        Parameters:
        - q: the query embeddings
        - X: the document embeddings
        '''
        
        # Send each tensors to device
        q.to(self.device)
        X.to(self.device)
    
        # Scaling factor for scaled dot product
        d = X.shape[-1]
        scale = 1 / math.sqrt(d)
        
        # Compute attention scores (scaled dot product)
        attn_scores = q @ X.transpose(-1, -2) * scale
        max_attn_scores = attn_scores.max(dim=-1).values
        attn_scores = max_attn_scores.mean(dim=-1)
        attn_scores.to(self.device)
    
        # Compute attention weights
        attn_weight = F.softmax(attn_scores, dim=-1)
        
        # Sort by score in descending order
        sorted_attn_score_args = attn_weight.argsort(descending=True)
    
        return sorted_attn_score_args

    def _retriever(self, query: str, k: int, verbose: bool) -> List[Document]:
        q, _ = self.get_embeddings(query)
        args = self.retrieve_with_attention(q, self.X).to('cpu').tolist()
        
        relevant_docs = [Document(page_content=self.docs[idx], id=str(idx)) for idx in args[:k]]
        # Display results
        if verbose:
            print(StringParser.from_langdocs(relevant_docs))
                
        return relevant_docs
                
    def as_retriever(self, search_kwargs: dict={'k': 4}, verbose=False) -> Runnable:
        if not isinstance(search_kwargs, dict):
            raise ValueError(f"search_kwargs must be a dict. Got {type(search_kwargs)}")

        if 'k' not in search_kwargs:
            search_kwargs['k'] = 4

        _retriever = functools.partial(self._retriever, k=search_kwargs['k'], verbose=verbose)

        return RunnableLambda(_retriever)