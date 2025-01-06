from langchain_core.messages import BaseMessage
from JB007.base.agent import Agent

from typing import List
from langchain import hub
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import BaseOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

from JB007.parsers.prompt import BasePromptParser, IdentityPromptParser

class EphemeralReactAgent(Agent):
    """React agent without memory."""
    def __init__(
            self, 
            name: str, 
            llm: BaseLLM | BaseChatModel, 
            tools: List[BaseTool],
            system_prompt = None,
            prompt_template: str = None, 
            prompt_parser:BasePromptParser = IdentityPromptParser(),
            output_parser:BaseOutputParser = None,
            verbose:bool = False
            ) -> None:
        
        super().__init__(name, llm=llm, prompt_template=prompt_template, system_prompt=system_prompt, prompt_parser=prompt_parser, output_parser=output_parser)
        self._tools = tools
        self._verbose = verbose
        # Init agent
        self._make_agent()

######################################## PRIVATE METHODS #########################################################

    def _make_agent(self):
        '''Create a react_agent using Langchain.'''
        # Sanity check
        if self._prompt_template is None and self._system_prompt is None:
            raise ValueError("Must have at least one of Union[system_prompt, prompt_template].")
        
        # To parse, or not to parse, that is the question
        if self._prompt_parser is None:
            self.prompt_parser = IdentityPromptParser()
            
        if self._output_parser is None:
            self.output_parser = RunnablePassthrough()

        ############## LLM Models #################### TODO:

        if isinstance(self._llm, BaseLLM):
            # Parse prompt template
            parsed_template = self.prompt_parser.parseSystemUser(self._system_prompt, self.prompt_template or hub.pull("hwchase17/react").template)

            # Define prompt
            prompt = PromptTemplate.from_template(parsed_template)
            
            # Make agentic chain
            self._agent = create_react_agent(self._llm, self._tools, prompt)
            
            # All done here...
            return
        
        ############## Chat Models ####################

        human_template = HumanMessagePromptTemplate.from_template(hub.pull("hwchase17/react").template)
        if self._prompt_template:
            # str template (text-only)
            if isinstance(self._prompt_template, str):
                human_template = HumanMessagePromptTemplate.from_template(self._prompt_template)
           
            # List[dict] support for multimodal template
            elif isinstance(self._prompt_template, list) and all(isinstance(item, dict) for item in self._prompt_template):
                # Add contents
                contents=[]
                for obj in self._prompt_template:
                    key = next(iter(obj))
                    if not key in self._supported_convo_keys:
                        raise ValueError(f"Unsupported key: Input keys shoud be one of Union['text', 'image_url']), but was given '{key}'")
                    contents.append({"type": key, f"{key}": obj[key]})
                human_template = HumanMessagePromptTemplate.from_template(contents)
            
            # Invalid format for prompt_template provided
            else:
                raise ValueError(f"Incorrect type for template: Should be one of Union[str, List[dict]]), but was given {type(input)}")
            
        # Define messages
        messages = [human_template, MessagesPlaceholder(variable_name='agent_scratchpad')]
        
        # Prepend system message if exists...
        if self._system_prompt:
            messages.insert(0, SystemMessage(content=self._system_prompt))
        
        # Build prompt
        prompt = ChatPromptTemplate.from_messages(messages)
        
        # Build agent
        self._agent = create_react_agent(self._llm, self._tools, prompt)

    def _invoke_with_prompt_template(self, input:str | dict | List[dict], config: RunnableConfig | None = None, stream:bool=False):    
        agent_executor = (
            AgentExecutor(agent=self._agent, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
            | RunnableLambda(lambda response: response["output"]) 
            | self._output_parser
        )
        
        # Input as a dictionary or string
        if isinstance(input, dict) or isinstance(input, str):
            # To stream, or not to stream, that is the question
            if stream:
                return agent_executor.stream(input, config)
            return agent_executor.invoke(input, config)

        # Input as a List[dict]
        if isinstance(input, list) and all(isinstance(element, dict) for element in input):
            chat_messages = self._compile_template_vars(input)

            # Ephemeral agentic chain
            anonymous_chain = create_react_agent(self._llm, self._tools, ChatPromptTemplate.from_messages(chat_messages))
            
            # Handle langchain side effect 
            if self._is_silent_caller:
                anonymous_chain = anonymous_chain | self._identity_fn
            
            # Ephemeral agent executor
            agent_executor = (
                AgentExecutor(agent=anonymous_chain, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
                | RunnableLambda(lambda response: response["output"]) 
                | self._output_parser
            )
            
            # To stream, or not to stream, that is the question
            if stream:
                return agent_executor.stream({}, config)
            return agent_executor.invoke({}, config)

        # Invalid input format     
        raise ValueError(f"Incorrect type fed to prompt_template: Should be one of Union[str, dict, List[dict]], but was given {type(input)}")
    
    def _invoke_without_prompt_template(self, input, config: RunnableConfig | None = None, stream:bool=False):
        messages = {"input": []}
        agent_executor = (
            AgentExecutor(agent=self._agent, tools=self._tools, verbose=self._verbose, handle_parsing_errors=True)
            | RunnableLambda(lambda response: response["output"]) 
            | self._output_parser
        )

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
            return agent_executor.stream(messages, config)
        return agent_executor.invoke(messages, config)

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
    
    def invoke(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: RunnableConfig | None = None):
        return super().invoke(input, config)
    
    def stream(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: RunnableConfig | None = None):
        return super().stream(input, config)
