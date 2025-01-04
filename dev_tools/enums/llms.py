import os

from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMs:
    @staticmethod
    def Phi3():
        phi = LlamaCpp(
            model_path=os.environ.get('MODEL_PATH_PHI_3_IT_Q4_GGUF'),
            n_gpu_layers=-1,
            n_batch=1024,
            n_ctx=4096,
            temperature=0,
            max_tokens=2000,
            f16_kv=True,
            verbose=False,
        )
        return phi
    
    @staticmethod
    def Phi35():
        phi = LlamaCpp(
            model_path=os.environ.get('MODEL_PATH_PHI_35_IT_Q4_GGUF'),
            n_gpu_layers=-1,
            n_batch=1024,
            n_ctx=8192,
            temperature=0,
            max_tokens=2000,
            f16_kv=True,
            verbose=False,
        )
        return phi
    

    @staticmethod
    def Gemma2Plus():
        google = LlamaCpp(
            model_path=os.environ.get('MODEL_PATH_GEMMA_9B_IT_Q4_GGUF'),
            n_gpu_layers=-1,
            n_batch=1024,
            n_ctx=8192,
            temperature=0,
            max_tokens=2000,
            f16_kv=True,
            verbose=False,
        )
        return google
    
    @staticmethod
    def Gemma2Mini():
        google = LlamaCpp(
            model_path=os.environ.get('MODEL_PATH_GEMMA_2B_IT_Q4_GGUF'),
            n_gpu_layers=-1,
            n_batch=1024,
            n_ctx=8192,
            temperature=0,
            max_tokens=2000,
            f16_kv=True,
            verbose=False,
        )
        return google
    
    @staticmethod
    def Gemini():
        gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        return gemini
    
    @staticmethod
    def Llama32():
        llama_3dot2 = LlamaCpp(
            model_path=os.environ.get('MODEL_PATH_LLAMA_32_IT_Q4_GGUF'),
            n_gpu_layers=-1,
            n_batch=1024,
            n_ctx=8192,
            temperature=0,
            max_tokens=2000,
            f16_kv=True,
            verbose=False,
        )
        return llama_3dot2