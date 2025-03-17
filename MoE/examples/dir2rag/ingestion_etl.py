import fitz  # PyMuPDF
from utils import Toc
from agents import section_summarizer
from agents import section_synth



class ETLDocIngestion:

    @staticmethod
    def extract_toc(pdf_path):
        """Extract the table of contents from a PDF document"""
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        toc_tree = Toc(toc)
        print(toc_tree.to_markdown())

        toc_tree.enhance_with_content(doc)
        print(toc_tree.to_markdown())

        toc_tree.summarize_toc(section_summarizer, section_synth)
        print(toc_tree.to_markdown())



