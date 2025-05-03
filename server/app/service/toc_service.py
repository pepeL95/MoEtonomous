from server.app.model.toc import ToC
from dev_tools.utils.pdf2md import Pdf2Markdown
from dev_tools.enums.llms import LLMs

class ToCService:
    @staticmethod    
    def get_toc(pdf_path):
        parser = Pdf2Markdown(pdf_path=pdf_path)
        toc = parser.get_toc(llm=LLMs.Gemini())
        return toc
        