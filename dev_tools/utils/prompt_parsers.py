class PromptParsers:
    class Gemma2:
        def parseSys(mssg):
            ret = (
                f"<start_of_turn>user\n"
                f"{mssg}<end_of_turn>\n"
            )
            return ret

        def parseUser(mssg):
            ret = (
                f"<start_of_turn>user\n"
                f"{mssg}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            return ret
        
        def parseTemplate(sys, usr):
            _sys = PromptParsers.Gemma2.parseSys(sys)
            _usr = PromptParsers.Gemma2.parseUser(usr)
            ret = _sys + _usr
            return ret
        
    class Phi35:
        def parseSys(mssg):
            ret = (
                f"<|system|>\n"
                f"{mssg}<|end|>\n"
            )
            return ret

        def parseUser(mssg):
            ret = (
                f"<|user|>\n"
                f"{mssg}<|end|>\n"
                f"<|assistant|>"
            )
            return ret
        
        def parseTemplate(sys, usr):
            _sys = PromptParsers.Phi35.parseSys(sys)
            _usr = PromptParsers.Phi35.parseUser(usr)
            ret = _sys + _usr
            return ret

    class LLama3:
        def parseSys(mssg):
            ret = (
                f"<|start_header_id|>system<|end_header_id|>\n"
                f"{mssg}\n"
                f"<|eot_id|>"
            )
            return ret
        
        def parseUser(mssg):
            ret = (
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"{mssg}\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
            return ret

        def parseTemplate(sys, usr):
            _sys = PromptParsers.LLama3.parseSys(sys)
            _usr = PromptParsers.LLama3.parseUser(usr)
            ret = _sys + _usr
            return ret

            