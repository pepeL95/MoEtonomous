import ast

from typing import List, Any
from xml.etree.ElementTree import fromstring

from langchain_core.output_parsers.base import BaseOutputParser

from agents.tools.toolschemas import Arxiv

class ArrayParser(BaseOutputParser[List[Any]]):
    '''Parse a string representation of a List of primitives into an array of primitives'''

    def parse(self, text):
        # Strip the surrounding code block markers if present
        if text.startswith("```"):
            code_block_start = text.find("[")
            code_block_end = text.rfind("]") + 1
            list_str = text[code_block_start:code_block_end]
        else:
            list_str = text

        # Parse the list using ast.literal_eval for safe evaluation
        try:
            parsed_list = ast.literal_eval(list_str)
            return parsed_list
        except (SyntaxError, ValueError):
            return []
