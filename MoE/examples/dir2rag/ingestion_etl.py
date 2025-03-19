import fitz  # PyMuPDF
from moe.examples.dir2rag.toc import Toc
from moe.examples.dir2rag.experts import section_summarizer, section_synth, metadata_xtractor
from langchain_core.documents import Document


class ETLDocIngestion:
    @staticmethod
    def extract_toc(pdf_path):
        """Extract the table of contents from a PDF document"""
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        toc_tree = Toc(toc)
        print(toc_tree.to_markdown())
        print(f'\n{30 * '-'}\n')

        toc_tree.enhance_with_content(doc)
        print(toc_tree.to_markdown())
        print(f'\n{30 * '-'}\n')

        toc_tree.summarize_toc(section_summarizer, section_synth)
        print(toc_tree.synthesis_to_markdown())
        print(f'\n{30 * '-'}\n')

        metadata = toc_tree.extract_metadata(doc, metadata_xtractor)
        print(metadata)
        print(f'\n{30 * '-'}\n')

        doc = Document(
            page_content=toc_tree.to_markdown(),
            metadata={
                'synthesis': toc_tree.synthesis_to_markdown(),
                'source': pdf_path,
                'title': metadata['title'],
                'authors': metadata['authors'],
                'date': metadata['date']
            }
        )

        return doc