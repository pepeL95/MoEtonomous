from abc import abstractmethod

from JB007.config.debug import Debug
from dev_tools.utils.clifont import CLIFont

class BasePromptParser:
    def __init__(self, verbosity:Debug.Verbosity=Debug.Verbosity.quiet):
        self.verbosity = verbosity

    @abstractmethod
    def parseSys(self, mssg:str):
        '''Parses system prompt into the right format for a given llm'''
        raise NotImplementedError("Subclass must implement abstract method")
    
    @abstractmethod
    def parseUser(self, mssg:str):
        '''Parses user prompt into the right format for a given llm'''
        raise NotImplementedError("Subclass must implement abstract method")
    
    @abstractmethod
    def parseSystemUser(self, sys:str, usr:str):
        raise NotImplementedError("Subclass must implement abstract method")

class IdentityPromptParser(BasePromptParser):
    def __init__(self, verbosity = Debug.Verbosity.quiet):
        super().__init__(verbosity)

    def parseSys(self, mssg:str):
        if self.verbosity != Debug.Verbosity.quiet:
            print(f'{CLIFont.bold}{CLIFont.light_green}{mssg}{CLIFont.reset}')
        return mssg

    def parseUser(self, mssg:str):
        if self.verbosity != Debug.Verbosity.quiet:
            print(f'{CLIFont.bold}{CLIFont.blue}{mssg}{CLIFont.reset}')
        return mssg

    def parseSystemUser(self, sys:str, usr:str):
        ret = sys + '\n\n' + usr
        if self.verbosity != Debug.Verbosity.quiet:
            print(f'{CLIFont.bold}{CLIFont.purple}{ret}{CLIFont.reset}')
        return ret