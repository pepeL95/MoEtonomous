from JB007.config.debug import Debug
from JB007.parsers.prompt import BasePromptParser

from dev_tools.utils.clifont import CLIFont

class PromptParsers:
    class Gemma2(BasePromptParser):
        def __init__(self, verbosity = Debug.Verbosity.quiet):
            super().__init__(verbosity)

        def parseSys(self, mssg):
            raise ValueError('The Gemma2 model does not allow system tags. Please include the system instructions in the prompt_template of your agent instead of in system_prompt.')

        def parseUser(self, mssg):
            ret = (
                f"<start_of_turn>user\n"
                f"{mssg}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.blue}{ret}{CLIFont.reset}')
                
            return ret
        
        def parseSystemUser(self, sys, usr):
            _sys = PromptParsers.Gemma2.parseSys(sys)
            _usr = PromptParsers.Gemma2.parseUser(usr)
            ret = _sys + _usr

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.purple}{ret}{CLIFont.reset}')

            return ret

    class Phi35(BasePromptParser):
        def __init__(self, verbosity = Debug.Verbosity.quiet):
            super().__init__(verbosity)

        def parseSys(self, mssg):
            ret = (
                f"<|system|>\n"
                f"{mssg}<|end|>\n"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.light_green}{ret}{CLIFont.reset}')

            return ret

        def parseUser(self, mssg):
            ret = (
                f"<|user|>\n"
                f"{mssg}<|end|>\n"
                f"<|assistant|>"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.blue}{ret}{CLIFont.reset}')

            return ret
        
        def parseSystemUser(self, sys, usr):
            _sys = PromptParsers.Phi35.parseSys(sys)
            _usr = PromptParsers.Phi35.parseUser(usr)
            ret = _sys + _usr

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.purple}{ret}{CLIFont.reset}')

            return ret

    class LLama32(BasePromptParser):
        def __init__(self, verbosity = Debug.Verbosity.quiet):
            super().__init__(verbosity)

        def parseSys(self, mssg):
            ret = (
                f"<|start_header_id|>system<|end_header_id|>\n"
                f"{mssg}\n"
                f"<|eot_id|>"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.light_green}{ret}{CLIFont.reset}')

            return ret
        
        def parseUser(self, mssg):
            ret = (
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"{mssg}\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.blue}{ret}{CLIFont.reset}')

            return ret

        def parseSystemUser(self, sys, usr):
            _sys = PromptParsers.LLama32.parseSys(sys)
            _usr = PromptParsers.LLama32.parseUser(usr)
            ret = _sys + _usr

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.purple}{ret}{CLIFont.reset}')

            return ret
