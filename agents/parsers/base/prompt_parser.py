from abc import abstractmethod

from agents.config.debug import Debug

class BasePromptParser:
    def __init__(self, verbosity: Debug.Verbosity = Debug.Verbosity.quiet):
        self.verbosity = verbosity

    @abstractmethod
    def parseSys(self, mssg: str):
        '''Parses system prompt into the right format for a given llm'''
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def parseUser(self, mssg: str):
        '''Parses user prompt into the right format for a given llm'''
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def parseSystemUser(self, mssgs: dict[str]):
        raise NotImplementedError("Subclass must implement abstract method")