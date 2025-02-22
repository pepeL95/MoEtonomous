from abc import abstractmethod
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableBinding
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser

from agents.parsers.base.prompt_parser import BasePromptParser
from agents.parsers.default.prompt_parser import DefaultPromptParser

class BaseAgent(Runnable):
    """Abstract class that offers a basic interface for specific agents"""

    def __init__(
        self,
        name: str,
        llm: BaseLLM | BaseChatModel = None,
        system_prompt: str = None,
        prompt_template: str = None,
        prompt_parser: BasePromptParser = DefaultPromptParser(),
        output_parser: BaseOutputParser | Runnable = StrOutputParser()
    ) -> None:

        self._name = name
        self._llm = llm
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template
        self._prompt_parser = prompt_parser
        self._output_parser = output_parser
        self._supported_convo_keys = set(["text", "image_url"])
        self._agent = None

################################################## GETTERS #####################################################

    @property
    def llm(self):
        return self._llm

    @property
    def prompt_template(self):
        return self._prompt_template

    @property
    def system_prompt(self):
        return self._system_prompt

    @property
    def output_parser(self):
        return self._output_parser

    @property
    def prompt_parser(self):
        return self._prompt_parser

    @property
    def name(self):
        return self._name

################################################## SETTERS #####################################################
    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(
                f'system_prompt must be of type str. Got {type(name)}')
        self._name = name

    @llm.setter
    def llm(self, llm: BaseLLM):
        if not isinstance(llm, (BaseLLM, BaseChatModel, RunnableBinding)):
            raise TypeError(
                f'llm must be of type Union[BaseLLM, BaseChatModel]. Got {type(llm)}')

        self._llm = llm
        self._make_agent()

    @system_prompt.setter
    def system_prompt(self, system_prompt: str):
        if not isinstance(system_prompt, str):
            raise TypeError(f'system_prompt must be of type str. Got {
                            type(system_prompt)}')

        self._system_prompt = self._prompt_parser.parseSys(system_prompt)
        self._make_agent()

    @prompt_template.setter
    def prompt_template(self, prompt_template: str):
        if not isinstance(prompt_template, str):
            raise TypeError(f'prompt_template must be of type str. Got {
                            type(prompt_template)}')

        self._prompt_template = self._prompt_parser.parseUser(prompt_template)
        self._make_agent()

    @prompt_parser.setter
    def prompt_parser(self, prompt_parser: BasePromptParser):
        if not isinstance(prompt_parser, BasePromptParser):
            raise TypeError(f'prompt_parser must be of type BasePromptParserGot {
                            type(prompt_parser)}')

        self._prompt_parser = prompt_parser
        self._make_agent()

    @output_parser.setter
    def output_parser(self, output_parser: BaseOutputParser):
        if not isinstance(output_parser, (BaseOutputParser, Runnable)):
            raise TypeError(f'output_parser must be of type Union[BaseOutputParser, Runnable]. Got {
                            type(output_parser)}')

        self._output_parser = output_parser
        self._make_agent()

################################################## ABSTRACT #####################################################

    @abstractmethod
    def _make_agent(self):
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def _invoke_with_prompt_template(self, input, stream):
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def _invoke_without_prompt_template(self, input, stream):
        raise NotImplementedError("Subclass must implement abstract method")

################################################## CONCRETE #####################################################

    def get_chain(self):
        return self._agent

    def invoke(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: Optional[RunnableConfig] = None):
        if not self._prompt_template:
            return self._invoke_without_prompt_template(input, config, stream=False)
        return self._invoke_with_prompt_template(input, config, stream=False)

    def stream(self, input: str | dict | List[dict] | BaseMessage | List[BaseMessage], config: Optional[RunnableConfig] = None):
        if not self._prompt_template:
            return self._invoke_without_prompt_template(input, config, stream=True)
        return self._invoke_with_prompt_template(input, config, stream=True)

    def _compile_user_ai_message(self, messages: list, entity: str = 'human'):
        # Sanity checks...
        if entity not in {'ai', 'human'}:
            raise ValueError(
                f'Entity should be one of [ai, human]. Got {entity}')

        if not all(isinstance(msg, dict) for msg in messages):
            raise ValueError(f'Messages should be a List[dict].')

        # Add contents
        contents = []
        for obj in messages:
            key = next(iter(obj))
            if not key in self._supported_convo_keys:
                raise ValueError(
                    f"Unsupported key: Input keys shoud be one of Union['text', 'image_url']), but was given '{key}'")
            contents.append({"type": key, f"{key}": obj[key]})

        # Create the message with contents
        return HumanMessage(content=contents, role='user') if entity == 'human' else AIMessage(content=contents, role='assistant')
