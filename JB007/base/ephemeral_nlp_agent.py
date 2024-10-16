from JB007.base.agent import Agent

from typing import Union, List

from langchain_community.llms import BaseLLM
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate

class EphemeralNLPAgent(Agent):
    '''Multimodal conversational agent without memory.'''
    def __init__(
            self, 
            name: str,
            llm: BaseLLM,
            system_prompt:str = None, 
            prompt_template:Union[str, List[dict]] = None, 
            parser:BaseOutputParser = StrOutputParser()
            ) -> None:
        
        super().__init__(name, llm=llm, system_prompt=system_prompt, prompt_template=prompt_template, parser=parser)
        self._supported_convo_keys = set(["text", "image_url"])
        # Init agent
        self._make_agent()

######################################## PRIVATE METHODS #########################################################

    def _make_agent(self):
        '''Conversational Multimodal Agent'''
        # Some sanity check...
        if self._system_prompt is None and self._prompt_template is None:
            raise ValueError("You must provide at least one of [system_prompt, prompt_template]")
        
        # Only a prompt template (i.e. no system prompt) was provided (this is the recommended way to use the agent for llms that require a niche chat template)
        if self._system_prompt is None and self._prompt_template is not None:
            # Define prompt
            prompt = PromptTemplate.from_template(self._prompt_template)

            # To parse, or not to parse, that is the question
            if self.parser is None:
                self.parser = RunnablePassthrough()
            
            # Run agent
            self._agent = prompt | self.llm | self._parser
            
            # All done here...
            return

        # Both, system prompt and prompt template were provided (this is the recommended way for well-established llms such as chatgpt, gemini, etc...)
        if template:= self._prompt_template:
            # str template (text-only)
            if isinstance(template, str):
                template = HumanMessagePromptTemplate.from_template(self._prompt_template)
           
            # List[str] template (support multimodal)
            elif isinstance(template, list) and all(isinstance(item, dict) for item in template):
                # Add contents
                contents=[]
                for obj in template:
                    key = next(iter(obj))
                    if not key in self._supported_convo_keys:
                        raise ValueError(f"Unsupported key: Input keys shoud be one of Union['text', 'image_url']), but was given '{key}'")
                    contents.append({"type": key, f"{key}": obj[key]})
                template = HumanMessagePromptTemplate.from_template(contents)
            
            # Invalid format for prompt_template provided
            else:
                raise ValueError(f"Incorrect type for template: Should be one of Union[str, List[dict]]), but was given {type(input)}")

        # Define prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self._system_prompt),
                template or MessagesPlaceholder(variable_name="input")
            ]
        )

        # To parse, or not to parse, that is the question
        if self.parser is None:
            self.parser = RunnablePassthrough()
        
        # Create chain
        self._agent = prompt | self.llm | self._parser

    def _invoke_with_prompt_template(self, input: Union[str, dict, List[dict]], stream=False):
        '''
        Invoke agent when a prompt template is defined.
        input should match the prompt template definition accordingly.
        '''
        # Input as a dict
        if isinstance(input, dict) or isinstance(input, str):
            if stream:
                return self._agent.stream(input)
            return self._agent.invoke(input)
        
        # Input as a List[dict]
        if isinstance(input, list) and all(isinstance(element, dict) for element in input):
            chat_messages = self._compile_template_vars(input)
            
            # To parse, or not to parse, that is the question
            if self.parser is None:
                self.parser = RunnablePassthrough()
            
            # Temporary update chat template
            anonymous_chain = ChatPromptTemplate.from_messages(chat_messages) | self.llm | self.parser
            
            # To stream, or not to stream, that is the question
            if stream:
                return anonymous_chain.stream({})
            return anonymous_chain.invoke({})

        # Invalid input format     
        raise ValueError(f"Incorrect type fed to prompt_template: Should be one of Union[str, dict, List[dict]], but was given {type(input)}")

    def _invoke_without_prompt_template(self, input: Union[str, dict, List[dict], BaseMessage, List[BaseMessage]], stream=False):
        messages = {"input": []}
        
        # Input as a str
        if isinstance(input, str):
            message = HumanMessage(content=input)
            messages["input"].append(message)
        
        # Input as a dict
        elif isinstance(input, dict):
            if 'input' not in input:
                raise ValueError("Missing 'input' key in your input object. Maybe you meant to provide a prompt_template before invoking?")
            message = HumanMessage(content=input['input'], role='user')
            messages['input'].append(message)
        
        # Input as a BaseMessage
        elif isinstance(input, BaseMessage):
            messages["input"].append(input)
        
        # Input as a List[BaseMessage]
        elif isinstance(input, list) and all(isinstance(item, BaseMessage) for item in input):
            messages["input"].extend(input)
        
        # Input as a List[dict]
        elif isinstance(input, list) and all(isinstance(item, dict) for item in input):
            message = self._compile_user_ai_message(entity='human', messages=input)
            messages["input"].append(message)
        
        # Invalid input format
        else:
            raise ValueError(f"Incorrect type for input_object: Should be one of Union[str, List[dict], List[BaseMessage], BaseMessage]), but was given {type(input)}")
        
        # To stream, or not to stream, that is the question
        if stream:
            return self._agent.stream(messages)
        return self._agent.invoke(messages)

############################################# CLASS PRIVATE METHODS ####################################################

    def _compile_user_ai_message(self, messages:list, entity:str='human'):
        # Sanity checks...
        if entity not in {'ai', 'human'}:
            raise ValueError(f'Entity should be one of [ai, human]. Got {entity}')
        
        if not all(isinstance(msg, dict) for msg in messages):
            raise ValueError(f'Messages should be a List[dict].')

        # Add contents
        contents=[]
        for obj in messages:
            key = next(iter(obj))
            if not key in self._supported_convo_keys:
                raise ValueError(f"Unsupported key: Input keys shoud be one of Union['text', 'image_url']), but was given '{key}'")
            contents.append({"type": key, f"{key}": obj[key]})
    
        # Create the message with contents
        return HumanMessage(content=contents, role='user') if entity == 'human' else AIMessage(content=contents, role='assistant')

    def _compile_template_vars(self, input):
        '''
        Compiles template_vars from input into a HumanMessage
        Returns: [
            ...previously defined BaseMessgaes at make_agent() time,
            HumanMessage(contents=[...])
        ]
        '''
        chat_messages = self._agent.first.messages.copy()
        contents = []
        # Iterate through input objects (e.g. [{'template_vars': [...]}, {'text': 'some text'}, ...])
        for obj in input:
            key = next(iter(obj))
            
            # Only valid keys allowed
            if not key in self._supported_convo_keys.union({'template_vars'}):
                raise ValueError(f"Unsupported key: Input keys shoud be one of Union['template_vars', 'text', 'image_url']), but was given '{key}'")
            
            # Extract template_vars
            if key == "template_vars":
                # {template_var_i_key: template_var_i_value, ...}
                kwargs = obj[key]
                chat_messages[-1] = chat_messages[-1].format(**kwargs)
                # Note: at this point messages[-1] is always a HumanMessage (since we formatted it)
                # Note: template_messages[-1].content can only be Union[str, List[dict]]
                content = chat_messages[-1].content 
                # Parse content to valid List[dict] (if necessary)
                if isinstance(content, str):
                    content = [{'type': 'text', 'text': content}]

                # Note: At this point all contents are List[dict]
                contents += content
            else:
                contents.append({"type": key, f"{key}": obj[key]})

        # If at this point chat_messages[-1] is a HumanMessagePromptTemplate still, convert it to HumanMessage
        if isinstance(chat_messages[-1], HumanMessagePromptTemplate):
            # But, we must make sure that it doesnt have input_variables. If it does, then the user didnt provide template_vars mistakenly
            if chat_messages[-1].input_variables:
                raise ValueError('You must provide the all required template_vars')
            # Otherwise, the prompt template was just plain text with no input variables, so use the template as text.
            content = [{'type': 'text', 'text': prompt.template} for prompt in chat_messages[-1].prompt]
            contents = content + contents

        # Update the messages
        chat_messages[-1] = HumanMessage(content=contents)
        return chat_messages

############################################# PUBLIC METHODS ####################################################


    def get_chain(self):
        return super().get_chain()
    
    def invoke(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage]):
        return super().invoke(input)
    
    def stream(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage]):
        return super().stream(input)
        