from JB007.toolbox.toolschemas import ToolSchemas

from langchain_core.output_parsers import JsonOutputParser

class ArxivPyParser:
    @staticmethod
    def apiQueryJson():
        return JsonOutputParser(pydantic_object=ToolSchemas.Arxiv.ApiQuery)
    
            
