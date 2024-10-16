from abc import abstractmethod

class Agent:
    """Abstract class that offers a basic interface for specific agents"""
    def __init__(self, name, llm=None, system_prompt=None, prompt_template=None, parser=None):
        self._name = name
        self._llm = llm
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template
        self._parser = parser
        self._agent = None

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
    def parser(self):
        return self._parser
    
    @property
    def name(self):
        return self._name
        
    def get_chain(self):
        return self._agent
    
    @abstractmethod
    def _make_agent(self):
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def _make_agentic_chain(self):
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def _get_agent_node(self):
        raise NotImplementedError("Subclass must implement abstract method")
        
    @abstractmethod
    def add_prompt_template(self, template):
        raise NotImplementedError("Subclass must implement abstract method")
    
    @abstractmethod
    def invoke(self, input_object):
        raise NotImplementedError("Subclass must implement abstract method")