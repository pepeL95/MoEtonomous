import os

from enum import Enum

from langchain_community.chat_models import ChatLlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI

from dev_tools.patterns.singleton import Singleton


class Gemini(Singleton):
    """Gemini model initialized as a singleton."""

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0)
        
class Phi3(Singleton):
    """Phi3 model initialized as a singleton."""

    def __init__(
            self, 
            n_gpu_layers=-1, 
            n_batch=1024, 
            n_ctx=8192, 
            temperature=0, 
            max_tokens=2000, 
            f16_kv=True, 
            verbose=False
        ) -> None:

        self.model = ChatLlamaCpp(
            model_path=os.environ.get('MODEL_PATH_PHI_3_IT_Q4_GGUF'),
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            temperature=temperature,
            max_tokens=max_tokens,
            f16_kv=f16_kv,
            verbose=verbose,
        )

class Phi35(Singleton):
    """Phi35 model initialized as a singleton."""

    def __init__(
            self, 
            n_gpu_layers=-1, 
            n_batch=1024, 
            n_ctx=8192, 
            temperature=0, 
            max_tokens=2000, 
            f16_kv=True, 
            verbose=False
        ) -> None:

        self.model = ChatLlamaCpp(
            model_path=os.environ.get('MODEL_PATH_PHI_35_IT_Q4_GGUF'),
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            temperature=temperature,
            max_tokens=max_tokens,
            f16_kv=f16_kv,
            verbose=verbose,
        )

class Gemma2Plus(Singleton):
    """Gemma2Plus model initialized as a singleton."""

    def __init__(
            self, 
            n_gpu_layers=-1, 
            n_batch=1024, 
            n_ctx=8192, 
            temperature=0, 
            max_tokens=2000, 
            f16_kv=True, 
            verbose=False
        ) -> None:

        self.model = ChatLlamaCpp(
            model_path=os.environ.get('MODEL_PATH_GEMMA_9B_IT_Q4_GGUF'),
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            temperature=temperature,
            max_tokens=max_tokens,
            f16_kv=f16_kv,
            verbose=verbose,
        )

class Gemma2Mini(Singleton):
    """Gemma2Mini model initialized as a singleton."""

    def __init__(
            self, 
            n_gpu_layers=-1, 
            n_batch=1024, 
            n_ctx=8192, 
            temperature=0, 
            max_tokens=2000, 
            f16_kv=True, 
            verbose=False
        ) -> None:

        self.model = ChatLlamaCpp(
            model_path=os.environ.get('MODEL_PATH_GEMMA_2B_IT_Q4_GGUF'),
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            temperature=temperature,
            max_tokens=max_tokens,
            f16_kv=f16_kv,
            verbose=verbose,
        )

class Llama32(Singleton):
    """Llama32 model initialized as a singleton."""

    def __init__(
            self, 
            n_gpu_layers=-1, 
            n_batch=1024, 
            n_ctx=8192, 
            temperature=0, 
            max_tokens=2000, 
            f16_kv=True, 
            verbose=False
        ) -> None:

        self.model = ChatLlamaCpp(
            model_path=os.environ.get('MODEL_PATH_LLAMA_32_IT_Q4_GGUF'),
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            temperature=temperature,
            max_tokens=max_tokens,
            f16_kv=f16_kv,
            verbose=verbose,
        )

class DeepSeekLlama(Singleton):
    """DeepSeekLlama model initialized as a singleton."""

    def __init__(
            self, 
            n_gpu_layers=-1, 
            n_batch=1024, 
            n_ctx=8192, 
            temperature=0, 
            max_tokens=2000, 
            f16_kv=True, 
            verbose=False
        ) -> None:

        self.model = ChatLlamaCpp(
            model_path=os.environ.get('MODEL_PATH_DEEPSEEK_LLAMA_Q4_GGUF'),
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            temperature=temperature,
            max_tokens=max_tokens,
            f16_kv=f16_kv,
            verbose=verbose,
        )
    
    
class LLMs(Enum):
    DeepSeekLlama = DeepSeekLlama
    Gemini = Gemini
    Phi3 = Phi3
    Phi35 = Phi35
    Gemma2Plus = Gemma2Plus
    Gemma2Mini = Gemma2Mini
    Llama32 = Llama32
    
    def __call__(self):
        return self.value().model
    
    
    
