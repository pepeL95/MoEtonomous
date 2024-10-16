from james_bond.agents.ephemeral_nlp_agent import EphemeralNLPAgent

from typing import List, Union
from langchain_community.llms import BaseLLM
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import BaseOutputParser

prompt_template = """\
You are given a passage. Your task is to extract the features according to the criteria given below.

### Passage
{passage}

### Criteria
The following criteria is defined as {{feature: feature description}}. Respond with a 0 for False or a 1 for True on each criterium.

{feature_catalogue}

### Output Format
Output JSON
"""

class FeatureVectorizerAgent(EphemeralNLPAgent):
    '''
    This agent extracts the features described in the feature_catalogue, and into a 
    vector f = [f_0, ..., f_n] where f_i is the ith feature explained by the feature_catalogue
    '''
    def __init__(self,
                name:str, 
                llm:BaseLLM, 
                feature_catalogue: Union[dict, BaseModel], # {feature_name: feature_description, ...}
                system_prompt:str="You are a feature extraction expert.",
                prompt_template:Union[str, List[dict]]=prompt_template,
                parser: BaseOutputParser=None,
                ) -> None:

        super().__init__(name, llm, system_prompt, prompt_template, parser)
        self._feature_catalogue = feature_catalogue
        
        if self._prompt_template is None:
            raise ValueError('Prompt template cannot be None because it must specify the feature_catalogue. Please, ensure you provide a prompt template with at least feature_catalogue as input variable.')
        
        self._make_agent()

        if 'feature_catalogue' not in self._agent.first.input_variables:
            raise ValueError('You must provide feature_catalogue as an input variable to the prompt template')

    def _make_agent(self):
        super()._make_agent()

    def invoke(self, input_object:Union[str, dict, List[dict], BaseMessage]):
        # Handle adding feature_catalogue to input_object
        if isinstance(input_object, str):
            input_key = 'input'
            for key in self._agent.first.input_variables:
                if key != 'feature_catalogue':
                    input_key = key
            input_object = {'feature_catalogue': self._feature_catalogue, input_key: input_object}
        
        elif isinstance(input_object, dict):
            input_object['feature_catalogue'] = self._feature_catalogue
        
        elif all([isinstance(input_object, List), input_object, isinstance(input_object[0], dict)]):
            has_template_vars = False
            for obj in input_object:
                key = next(iter(obj))
                if key == 'template_vars':
                    assert isinstance(obj[key], dict)
                    has_template_vars = True
                    obj[key]['feature_catalogue'] = self._feature_catalogue
                if has_template_vars:
                    break
            
            if not has_template_vars:
                input_object = [{'template_vars': {'feature_catalogue': self._feature_catalogue}}] + input_object

        # Call agent
        return super().invoke(input_object)
