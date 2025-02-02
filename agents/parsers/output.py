from agents.toolbox.toolschemas import ToolSchemas

import ast
from typing import List
import xml.etree.ElementTree as ET

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser

        
class StringParser:
    @staticmethod
    def from_langdocs(docs: List[Document]) -> str:
        if docs and isinstance(docs[0], Document):
            return '\n\n'.join(doc.page_content for doc in docs)
        return '\n\n'.join(docs)

    @staticmethod
    def from_array(docs: List[str]):
        return '\n\n'.join(docs)

    @staticmethod
    def to_array(input_str):
        # Strip the surrounding code block markers if present
        if input_str.startswith("```"):
            code_block_start = input_str.find("['")
            code_block_end = input_str.rfind("']") + 2
            list_str = input_str[code_block_start:code_block_end]
        else:
            list_str = input_str

        # Parse the list using ast.literal_eval for safe evaluation
        try:
            parsed_list = ast.literal_eval(list_str)
            return parsed_list
        except (SyntaxError, ValueError):
            return []


class ArxivParser:
    class ApiSearchItems:
        @staticmethod
        def to_json():
            return JsonOutputParser(pydantic_object=ToolSchemas.Arxiv.ApiSearchItems)

    class XML:
        @staticmethod
        def to_dict(xml_string: str) -> dict:
            # Parse XML string
            root = ET.fromstring(xml_string)
            # Define namespace
            namespace = {"atom": "http://www.w3.org/2005/Atom"}

            # Find entries
            entries = root.findall('atom:entry', namespace)

            # Initialize array to store dictionary objects
            articles = []
            # Iterate over each entry
            for entry in entries:
                article = {
                    "title": entry.find('atom:title', namespace).text.replace('\n', '').strip(),
                    "published_date": entry.find('atom:updated', namespace).text.strip(),
                    "pdf_link": entry.find("atom:link[@title='pdf']", namespace).attrib['href'],
                    "abstract": entry.find('atom:summary', namespace).text.strip()
                }
                articles.append(article)
            return articles
