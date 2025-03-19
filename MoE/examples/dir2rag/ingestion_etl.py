############################# SYS PATH LOAD ################################

from dotenv import load_dotenv
import sys
import os


if not os.environ.get('ENV'):
    print('Setting up env')
    load_dotenv(os.environ["RND_ENV_CONFIG_PATH"])  # .env file path
    sys.path.append(os.environ.get('SRC'))

###########################################################################

import fitz  # PyMuPDF
from moe.examples.dir2rag.toc import Toc
from moe.examples.dir2rag.experts import section_summarizer, section_synth



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



if __name__ == "__main__":
    etl = ETLDocIngestion.extract_toc("/Users/pepelopez/Documents/Learning/Genai/Papers/openelm.pdf")
    etl.toc_tree.to_markdown()