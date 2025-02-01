from JB007.config.debug import Debug
from JB007.parsers.prompt import BasePromptParser

from dev_tools.utils.clifont import CLIFont


class PromptParsers:
    class Identity(BasePromptParser):
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

    class Gemma2(BasePromptParser):
        def __init__(self, verbosity=Debug.Verbosity.quiet):
            super().__init__(verbosity)

        def parseSys(self, mssg):
            if mssg is None:
                return None
            raise ValueError("Gemma2 does not support system tags")

        def parseUser(self, mssg):
            if mssg is None:
                return None

            ret = (
                f"<start_of_turn>user\n"
                f"{mssg}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.blue}{ret}{CLIFont.reset}')

            return ret

        def parseSystemUser(self, sys, usr):
            _sys = sys or ""
            _usr = usr or "{input}"

            ret = self.parseUser(_sys + '\n' + _usr)

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.purple}{ret}{CLIFont.reset}')

            return ret

    class Phi35(BasePromptParser):
        def __init__(self, verbosity=Debug.Verbosity.quiet):
            super().__init__(verbosity)

        def parseSys(self, mssg):
            if mssg is None:
                return None

            ret = (
                f"<|system|>\n"
                f"{mssg}<|end|>\n"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.light_green}{ret}{CLIFont.reset}')

            return ret

        def parseUser(self, mssg):
            if mssg is None:
                return None

            ret = (
                f"<|user|>\n"
                f"{mssg}<|end|>\n"
                f"<|assistant|>"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.blue}{ret}{CLIFont.reset}')

            return ret

        def parseSystemUser(self, sys, usr):
            _sys, _usr = "", "{input}"

            if sys:
                _sys = self.parseSys(sys)
            if usr:
                _usr = self.parseUser(usr)

            ret = _sys + _usr

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.purple}{ret}{CLIFont.reset}')

            return ret

    class LLama32(BasePromptParser):
        def __init__(self, verbosity=Debug.Verbosity.quiet):
            super().__init__(verbosity)

        def parseSys(self, mssg):
            if mssg is None:
                return None

            ret = (
                f"<|start_header_id|>system<|end_header_id|>\n"
                f"{mssg}\n"
                f"<|eot_id|>"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.light_green}{
                      ret}{CLIFont.reset}')

            return ret

        def parseUser(self, mssg):
            if mssg is None:
                return None

            ret = (
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"{mssg}\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.blue}{ret}{CLIFont.reset}')

            return ret

        def parseSystemUser(self, sys, usr):
            _sys, _usr = "", "{input}"

            if sys:
                _sys = self.parseSys(sys)
            if usr:
                _usr = self.parseUser(usr)

            ret = _sys + _usr

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.purple}{ret}{CLIFont.reset}')

            return ret
    
    class DeepSeekLLama(BasePromptParser):
        def __init__(self, verbosity=Debug.Verbosity.quiet):
            super().__init__(verbosity)

        def parseSys(self, mssg):
            if mssg is None:
                return None

            ret = (
                f"<|begin▁of▁sentence|>{mssg}"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.light_green}{ret}{CLIFont.reset}')

            return ret

        def parseUser(self, mssg):
            if mssg is None:
                return None

            ret = (
                f"<|User|>{mssg}<|Assistant|>"
            )

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.blue}{ret}{CLIFont.reset}')

            return ret

        def parseSystemUser(self, sys, usr):
            _sys, _usr = "", "{input}"

            if sys:
                _sys = self.parseSys(sys)
            if usr:
                _usr = self.parseUser(usr)

            ret = _sys + _usr

            if self.verbosity != Debug.Verbosity.quiet:
                print(f'{CLIFont.bold}{CLIFont.purple}{ret}{CLIFont.reset}')

            return ret
