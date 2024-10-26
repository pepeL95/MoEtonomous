from JB007.base.ephemeral_nlp_agent import EphemeralNLPAgent

from typing import List

from langchain_community.llms import BaseLLM
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate


class PersistentNLPAgent(EphemeralNLPAgent):
    '''Multimodal conversational agent without memory.'''
    def __init__(
            self, 
            name: str, 
            llm: BaseLLM, 
            system_prompt: str = None, 
            prompt_template: str = None, 
            parser: BaseOutputParser = StrOutputParser()
            ) -> None:
        super().__init__(name, llm, system_prompt, prompt_template, parser)
        self._supported_convo_keys = set(["text", "image_url"])

######################################## CLASS METHODS #########################################################

    @classmethod
    def _validate_chat_history_is_list_base_messages(cls, chat_history):
        if not isinstance(chat_history, list):
            raise ValueError('chat_history must be a List[BaseMessage]')
        if not all(isinstance(msg, BaseMessage) for msg in chat_history):
            raise ValueError('chat_history must be a List[BaseMessage]')
        
    @classmethod
    def _extract_chat_history(cls, input: List[dict]):
        for obj in input:
            key = next(iter(obj))
            if key == 'chat_history':
                return obj[key]
        return []
    
    @classmethod
    def _compile_template_vars(cls, input):
            # Extract chat_history
            if chat_history := cls._extract_chat_history(input):
                cls._validate_chat_history_is_list_base_messages(chat_history)

            # Get formatted messages
            filtered_input = [input_obj for input_obj in input if 'chat_history' not in input_obj]
            chat_messages = super()._compile_template_vars(filtered_input)
            
            # Note: chat_messages[1] gives precisely the chat_history template (see make_agent())
            chat_messages[1] = chat_messages[1].format_messages(chat_history=chat_history)
            ret = chat_messages[:1] + chat_messages[1] + chat_messages[2:]
            return ret
    

######################################## PRIVATE METHODS #########################################################


    def _make_agent(self):
        '''Multimodal conversational agent using Langchain.'''
        if self._system_prompt is None and self._prompt_template is None:
            raise ValueError("You must provide at least one of [system_prompt, prompt_template]")
        
        # Only a prompt template provided (i.e. no system_prompt)
        if self._system_prompt is None and self._prompt_template is not None:
            prompt = PromptTemplate.from_template(self._prompt_template)

            # To parse, or not to parse, that is the question
            if self.parser is None:
                self.parser = RunnablePassthrough()

            # Create chain
            self._agent = prompt | self.llm | self._parser

            # All done here...
            return

        # Both system_prompt and prompt_template provided
        if template:= self._prompt_template:
            # str-only template
            if isinstance(template, str):
                template = HumanMessagePromptTemplate.from_template(self._prompt_template)
                assert 'chat_history' not in template.input_variables, "chat_history is a reserved key for Persistent Agents, use Ephemeral"

            # Allows multimodal template
            elif isinstance(template, list) and all(isinstance(item, dict) for item in template):
                # Add contents
                contents=[]
                for obj in template:
                    key = next(iter(obj))
                    if not key in self._supported_convo_keys.difference('chat_history'):
                        raise ValueError(f"Unsupported key: Input keys shoud be one of Union['text', 'image_url']), but was given '{key}'")
                    contents.append({"type": key, f"{key}": obj[key]})
                template = HumanMessagePromptTemplate.from_template(contents)
                assert 'chat_history' not in template.input_variables, "chat_history is a reserved key for Persistent Agents, use Ephemeral"

            # Invalid format for prompt_template provided   
            else:
                raise ValueError(f"Incorrect type for template: Should be one of Union[str, List[dict]]), but was given {type(input)}")

        # Define prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self._system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                template or MessagesPlaceholder(variable_name="input")
            ]
        )

        # To parse, or not to parse, that is the question
        if self.parser is None:
            self.parser = RunnablePassthrough()

        self._agent = prompt | self.llm | self._parser

    def _invoke_with_prompt_template(self, input: str | dict | List[dict], stream=False):
        # Input as a dict
        if isinstance(input, dict):
            # Sanity check: chat_template should be a List[BaseMessages]
            if chat_history := input.get('chat_history'):
                self._validate_chat_history_is_list_base_messages(chat_history)
            
            # To stream, or not to stream, that is the question
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
            anonymous_chain = ChatPromptTemplate.from_messages(chat_messages) | self.llm | self._parser

            # To stream, or not to stream, that is the question
            if stream:
                return anonymous_chain.stream({})
            return anonymous_chain.invoke({})

        # Invalid format for input provided
        else:
            raise ValueError(f"input should be one of Union[dict, List[dict]]. Got {type(input)}")
    
    def _invoke_without_prompt_template(self, input: List[BaseMessage] | List[dict] | dict | str, stream=False):
        input_object = {"input": [], "chat_history": []}
        # Input as a str (i.e. no chat history)
        if isinstance(input, str):
            message = HumanMessage(content=input, role='user')
            input_object['input'].append(message)

        # Input as a dict (recall there is no template, so we must enforce the 'input' hey)
        elif isinstance(input, dict):
            # Sanity check: 'input' key?
            if 'input' not in input:
                raise ValueError("Missing required 'input' key in your input object.")
            
            # Sanity check: chat_template should be a List[BaseMessages]
            if chat_history := input.get('chat_history', []):
                self._validate_chat_history_is_list_base_messages(chat_history)

            # Build chat history and user input
            input_object['chat_history'] = chat_history
            input_object['input'].append(HumanMessage(content=input['input'], role='user'))

        # Input as a List[BaseMessage]   
        elif isinstance(input, List) and all(isinstance(item, BaseMessage) for item in input):
            input_object['chat_history'] += input[:-1] 
            input_object["input"].append(input[-1])

        # Input as a List[dict]
        elif isinstance(input, List) and all(isinstance(element, dict) for element in input):
            user_messages = []
            
            for message_obj in input:
                # Extract user_messages
                if 'chat_history' not in message_obj:
                    user_messages.append(message_obj)

                # Extract chat_history
                else:
                    for chat_history_msg in  message_obj['chat_history']:
                        # Sample chat_history_msg = {'ai': [{'text': 'some text'}, ...]}
                        entity = next(iter(chat_history_msg)) # human / ai
                        message = self._compile_user_ai_message(entity=entity, messages=chat_history_msg[entity])
                        input_object['chat_history'].append(message)
                          
            # Extract user prompt
            message = self._compile_user_ai_message(entity='human', messages=user_messages)
            input_object['input'].append(message)
        
        # Invalid format for input provided
        else:
            raise ValueError(f"Incorrect type for input_object: Should be one of Union[List[BaseMessage], List[dict]]), but was given {type(input)}")
        
        # To stream, or not to stream, that is the question
        if stream:
            return self._agent.stream(input_object)
        return self._agent.invoke(input_object)

############################################# PUBLIC METHODS ####################################################
    
    def invoke(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: RunnableConfig | None = None):
        return super().invoke(input, config)
    
    def stream(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: RunnableConfig | None = None):
        return super().stream(input, config)
    
    def get_chain(self):
        return super().get_chain()