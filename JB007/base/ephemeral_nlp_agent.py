from JB007.base.agent import Agent
from JB007.parsers.prompt import IdentityPromptParser, BasePromptParser

from typing import Union, List

from langchain_core.messages.base import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate


class EphemeralNLPAgent(Agent):
    '''Multimodal conversational agent without memory.'''

    def __init__(
            self,
            name: str,
            llm: BaseLLM | BaseChatModel,
            system_prompt: str = None,
            prompt_template: Union[str, List[dict]] = None,
            prompt_parser: BasePromptParser = IdentityPromptParser(),
            output_parser: BaseOutputParser = StrOutputParser()
    ) -> None:

        super().__init__(name, llm=llm, system_prompt=system_prompt,
                         prompt_template=prompt_template, prompt_parser=prompt_parser, output_parser=output_parser)
        self._supported_convo_keys = set(["text", "image_url"])
        # Init agent
        self._make_agent()

######################################## PRIVATE METHODS #########################################################

    def _make_agent(self):
        '''Conversational Multimodal Agent'''
        # Some sanity check...
        if self._system_prompt is None and self._prompt_template is None:
            raise ValueError(
                "You must provide at least one of [system_prompt, prompt_template]")

        # To parse, or not to parse, that is the question
        if self._prompt_parser is None:
            self.prompt_parser = IdentityPromptParser()

        if self._output_parser is None:
            self._output_parser = RunnablePassthrough()

        ############## LLM Models ####################

        if isinstance(self._llm, BaseLLM):
            # Prepare prompt template
            parsed_template = self.prompt_parser.parseSystemUser(
                self._system_prompt, self.prompt_template or "{input}")
            print(parsed_template)
            # Define prompt
            prompt = PromptTemplate.from_template(parsed_template)

            # Run agent
            self._agent = prompt | self.llm | self._output_parser

            # All done here...
            return

        ############## Chat Models ####################

        # Build user template
        human_template = HumanMessagePromptTemplate.from_template(
            self._prompt_parser.parseUser(self._prompt_template or "{input}"))

        # Define messages
        messages = [human_template]

        # Prepend system message if exists...
        if self._system_prompt:
            messages.insert(0, SystemMessagePromptTemplate.from_template(
                self._prompt_parser.parseSys(self._system_prompt)))

        # Build prompt
        prompt = ChatPromptTemplate.from_messages(messages)

        # Create chain
        self._agent = prompt | self._llm | self._output_parser

    def _invoke_with_prompt_template(self, input: Union[str, dict, List[dict]], config: RunnableConfig | None = None, stream: bool = False):
        '''
        Invoke agent when a prompt template is defined.
        input should match the prompt template definition accordingly.
        '''
        # Input as a dictionary or string
        if isinstance(input, dict) or isinstance(input, str):
            # To stream, or not to stream, that is the question
            if stream:
                return self._agent.stream(input, config)
            return self._agent.invoke(input, config)

        # Input as a List[dict]
        if isinstance(input, list) and all(isinstance(element, dict) for element in input):
            chat_messages = self._compile_template_vars(input)

            # Ephemeral chain
            anonymous_chain = ChatPromptTemplate.from_messages(
                chat_messages) | self.llm | self._output_parser

            # To stream, or not to stream, that is the question
            if stream:
                return anonymous_chain.stream({}, config)
            return anonymous_chain.invoke({}, config)

        # Invalid input format
        raise ValueError(
            f"Incorrect type fed to prompt_template: Should be one of Union[str, dict, List[dict]], but was given {type(input)}")

    def _invoke_without_prompt_template(self, input: Union[str, dict, List[dict], BaseMessage, List[BaseMessage]], config: RunnableConfig | None = None, stream=False):
        # Must have a system prompt
        messages = {"input": []}

        # Input as a str
        if isinstance(input, str):
            messages["input"] = input

        # Input as a dict
        elif isinstance(input, dict):
            if 'input' not in input:
                raise ValueError(
                    "Missing 'input' key in your input object. Maybe you meant to provide a prompt_template before invoking?")
            messages = input

        # Input as a BaseMessage
        elif isinstance(input, BaseMessage):
            messages["input"].append(input)

        # Input as a List[BaseMessage]
        elif isinstance(input, list) and all(isinstance(item, BaseMessage) for item in input):
            messages["input"].extend(input)

        # Input as a List[dict]
        elif isinstance(input, list) and all(isinstance(item, dict) for item in input):
            message = self._compile_user_ai_message(
                entity='human', messages=input)
            messages["input"].append(message)

        # Invalid input format
        else:
            raise ValueError(f"Incorrect type for input_object: Should be one of Union[str, List[dict], List[BaseMessage], BaseMessage]), but was given {
                             type(input)}")

        # To stream, or not to stream, that is the question
        if stream:
            return self._agent.stream(messages, config)
        return self._agent.invoke(messages, config)

############################################# CLASS PRIVATE METHODS ####################################################

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
                raise ValueError(
                    f"Unsupported key: Input keys shoud be one of Union['template_vars', 'text', 'image_url']), but was given '{key}'")

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
                raise ValueError(
                    'You must provide the all required template_vars')
            # Otherwise, the prompt template was just plain text with no input variables, so use the template as text.
            content = [{'type': 'text', 'text': prompt.template}
                       for prompt in chat_messages[-1].prompt]
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
