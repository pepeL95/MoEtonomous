from typing import List

from dev_tools.utils.clifont import CLIFont, print_bold

from agents.parsers.generic import StringParser
from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class MarkdownTree:
    @staticmethod
    def from_str(md_str: str):
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "Title"),
            ("##", "Section"),
            ("###", "Subsection"),
            ("####", "Detailed Subsection"),
        ])

        md_splits = md_splitter.split_text(md_str)
        if not 'Title' in md_splits[0].metadata:
            md_splits = md_splitter.split_text(f"# Document\n\n{md_str}")
        
        tree = MarkdownTree()
        tree._build_document_tree(md_chunks=md_splits)
        return tree
    
    @staticmethod
    def from_langdocs(md_chunks: List[Document]):
        tree = MarkdownTree()
        tree._build_document_tree(md_chunks=md_chunks)
        return tree

    def _build_document_tree(self, md_chunks: List[Document]):
        '''Builds a document hierarchy'''

        # Set root
        self.root = md_chunks[0]
        self.root.metadata['summary'] = None
        self.root.metadata['id'] = self.root.metadata['Title']

        # Vertices of the tree
        self.V = {self.root.metadata["id"]: self.root}

        for doc_split in md_chunks:
            _self, parent, grandparent = None, None, None

            # Create children if not exists
            if not "children" in doc_split.metadata:
                doc_split.metadata["children"] = []

            # <H4/>
            if "Detailed Subsection" in doc_split.metadata:
                _self = "Detailed Subsection"
                parent = "Subsection"
                grandparent = "Section"

            # <H3/>
            elif "Subsection" in doc_split.metadata:
                _self = "Subsection"
                parent = "Section"
                grandparent = "Title"

            # <H2/>
            elif "Section" in doc_split.metadata:
                _self = "Section"
                parent = "Title"
                grandparent = None

            # <H1/> (already set)
            elif "Title" in doc_split.metadata:
                continue

            # Otherwise, error
            else:
                raise ValueError(f"Metadata headers should be one of Union[Title, Section, Subsection, Detailed Subsection]. Got `{doc_split.metadata}` instead.")

            # Enhance doc_split metadata
            doc_split.metadata["id"] = doc_split.metadata.get(_self)
            doc_split.metadata["summary"] = None
            
            # Add doc_split to V set
            self.V[doc_split.metadata["id"]] = doc_split

            # Add parent if not exists
            if (parent_id := doc_split.metadata.get(parent)) not in self.V:
                self.V[parent_id] = Document(
                    page_content="This is a high-level vertex",
                    metadata={
                        "children": [],
                        "summary": None,
                        "id": parent_id,
                        "Title": self.root.metadata.get("Title"),
                        "Section": doc_split.metadata.get("Section"),
                        _self == "Detailed Subsection" and "Subsection": doc_split.metadata.get("Subsection"),
                    }
                )

                # Update grandparent
                if (grandparent_id := doc_split.metadata.get(grandparent)):
                    self.V[grandparent_id].metadata['children'].append(self.V[parent_id]['id'])
            
            # Update parent
            self.V[parent_id].metadata['children'].append(doc_split.metadata['id'])

    def create_summaries_recursively(self, agent: Runnable):
        def _run_recursion(curr=self.root, agent=agent):
            # Base case
            if not curr.metadata['children']:
                curr.metadata['summary'] = agent.invoke({'input': curr.page_content})
                return

            # Recursion
            for child in curr.metadata['children']:
                _run_recursion(curr=self.V[child], agent=agent)
            
            # Generate parent summary
            summaries = [self.V[child].metadata['summary'] for child in curr.metadata['children'] if self.V[child].metadata.get('summary')]
            curr.metadata['summary'] = agent.invoke({
                'input': StringParser.from_array(summaries)
            })

        _run_recursion()

    def flatten_tree(self):
        flat_tree = []
        queue = [self.root]
        while queue:
            current = queue.pop(0)
            flat_tree.append(current)
            queue += [self.V[child] for child in current.metadata["children"]]
        return flat_tree

    def print_tree(self):
        """Pre-order print"""
        queue = [self.root]
        while queue:
            current = queue.pop(0)

            title = current.metadata.get("Title")
            section = current.metadata.get("Section")
            subsection = current.metadata.get("Subsection")
            detailed_subsection = current.metadata.get("Detailed Subsection")

            print()
            if title:
                print_bold(f"{CLIFont.light_green}{title}", end='')
            if section:
                print_bold(f" | {CLIFont.light_green}{section}", end='')
            if subsection:
                print_bold(f" | {CLIFont.light_green}{subsection}", end='')
            if detailed_subsection:
                print_bold(
                    f" | {CLIFont.light_green}{detailed_subsection}", end='')
            print()

            print_bold('Page Content:')
            print(current.page_content)
            if current.metadata.get("summary"):
                print_bold("Summary:")
                print(current.metadata.get("summary"))
            print(100 * '*')
            queue += [self.V[child] for child in current.metadata["children"]]