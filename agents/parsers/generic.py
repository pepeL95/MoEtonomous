from typing import List
from langchain_core.documents import Document

from xml.etree.ElementTree import fromstring
from agents.tools.toolschemas import ToolSchemas
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


class ArxivParser:
    class ApiSearchItems:
        @staticmethod
        def to_json():
            return JsonOutputParser(pydantic_object=ToolSchemas.Arxiv.ApiSearchItems)

    class XML:
        @staticmethod
        def to_dict(xml_string: str) -> dict:
            # Parse XML string
            root = fromstring(xml_string)
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
