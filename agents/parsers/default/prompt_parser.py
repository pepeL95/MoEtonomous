from dev_tools.utils.clifont import CLIFont

from agents.config.debug import Debug
from agents.parsers.base.prompt_parser import BasePromptParser


class DefaultPromptParser(BasePromptParser):
    def __init__(self, verbosity=Debug.Verbosity.quiet):
        super().__init__(verbosity)

    def parseSys(self, mssg: str):
        if self.verbosity != Debug.Verbosity.quiet:
            print(f'{CLIFont.bold}{CLIFont.light_green}{mssg}{CLIFont.reset}')
        return mssg

    def parseUser(self, mssg: str):
        if self.verbosity != Debug.Verbosity.quiet:
            print(f'{CLIFont.bold}{CLIFont.blue}{mssg}{CLIFont.reset}')
        return mssg

    def parseSystemUser(self, sys: str, usr: str):
        _sys, _usr = "", "{input}"

        if sys:
            _sys = self.parseSys(sys) + '\n\n'
        if usr:
            _usr = self.parseUser(usr)

        ret = _sys + _usr
        if self.verbosity != Debug.Verbosity.quiet:
            print(f'{CLIFont.bold}{CLIFont.purple}{ret}{CLIFont.reset}')
        return ret
