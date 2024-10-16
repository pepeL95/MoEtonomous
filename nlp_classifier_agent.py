from rag.utils.ragformer import RagFormer
from james_bond.agents.agent import Agent

from typing import Union, List, Dict
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings.embeddings import Embeddings
from transformers.models.mpnet.modeling_mpnet import MPNetModel
from transformers.models.mpnet.tokenization_mpnet_fast import MPNetTokenizerFast

DEFAULT_CLASSIFIER_TEMPLATE = """\
You are an expert in categorical classification.

## Classes (categories):
{class_names}

## Descriptions:
{class_descriptions}

## Task:
- Classify the user input into one of the categories based on the descriptions.
- Provide a one-sentence reasoning for your choice.
- Do not answer the user input; only classify.

## User Input:
{classifier_input}

## Output:
Classifier: """

class NLPClassifierAgent(Agent):
    '''
    The way the NLPClassifier works is as follows:
    - It asks the llm to classify an input into one of the provided classes (following the provided descriptions). 
    - We ask the LLM to also provide its reasoning of why it chose thus class.This aims to enhance accuracy following soa prompt engineering (e.g. CoT).
    - We finally obtain the output from the LLM and run sentence similarity with the class descriptions to make sure we output one of the valid classes.
    '''
    def __init__(
            self, 
            name: str, 
            llm: BaseLLM,
            embedding_model: Union[Embeddings, MPNetModel],
            class_names: List[str],
            class_descriptions: List[str],
            tokenizer: MPNetTokenizerFast = None,
            prompt_template: str = DEFAULT_CLASSIFIER_TEMPLATE,
            verbose:bool = True,
        ) -> None:
        
        super().__init__(name=name, llm=llm, prompt_template=prompt_template, parser=StrOutputParser())
        self._class_names = class_names
        self._class_descriptions = class_descriptions
        self._embedding_model = embedding_model
        self._tokenizer = tokenizer
        self._verbose = verbose
        # make classifier
        self._make_agent()
    
    def _make_agent(self) -> None:
        vectorsearch = None
        # When we use a huggingface embedding model, we use RagFormer
        if isinstance(self._embedding_model, MPNetModel):
            # Ensure we also have the corresponding tokenizer (this is only needed for HF embeddings)
            if self._tokenizer is None:
                raise ValueError("This embedding model needs a tokenizer. Please provide one of type MPNetTokenizerFast")
            vectorsearch = RagFormer(embedding_model=self._embedding_model, tokenizer=self._tokenizer, docs=self._class_names)
        
        # Otherwise, for use with Embeddings (e.g. OpenAIEmbeddings), we use FAISS for similarity
        elif isinstance(self._embedding_model, Embeddings):
            vectorsearch = FAISS.from_texts(texts=self._class_names, embedding=self._embedding_model, ids=[str(i) for i in range(len(self._class_names))])
        
        # Not a supported embedding model
        else:
            raise ValueError(f"Not a valid embedding model provided. You must provide one of Union[Embeddings, MPNetModel]. Got {type(self._embedding_model)} instead.")

        # Init chain
        retriever = vectorsearch.as_retriever(search_kwargs={'k': 1})
        prompt = PromptTemplate.from_template(self._prompt_template)
        self._agent = (prompt | self._llm | self._parser | retriever)
    
    def get_chain(self) ->  Runnable:
        return super().get_chain()
    
    def invoke(self, input: Union[str, Dict]) -> str:
        # Augment the input
        if isinstance(input, str):
            input = {
                "classifier_input": input,
                "class_names": self._class_names,
                "class_descriptions": self._class_descriptions
            }
        
        # Augment the input
        elif isinstance(input, dict):
            input["class_names"] = self._class_names,
            input["class_descriptions"] = self._class_descriptions

        # Return the closer class description according to the LLM prediction
        results = self._agent.invoke(input)

        if not results:
            return 'No class was selected.'
        
        class_name = self._class_names[int(results[0].id)]
        return class_name

        # return results
