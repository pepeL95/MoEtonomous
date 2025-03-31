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
        self.root.metadata['parent'] = None
        self.root.metadata["level"] = 1
        self.root.metadata['summary'] = None
        self.root.metadata['id'] = self.root.metadata['Title']

        # Vertices of the tree
        self.V = {self.root.metadata["id"]: self.root}

        for current_document in md_chunks:
            _self, parent, grandparent = None, None, None

            # Create children if not exists
            if not "children" in current_document.metadata:
                current_document.metadata["children"] = []

            # <H4/>
            if "Detailed Subsection" in current_document.metadata:
                _self = "Detailed Subsection"
                parent = "Subsection"
                grandparent = "Section"
                current_document.metadata["level"] = 4

            # <H3/>
            elif "Subsection" in current_document.metadata:
                _self = "Subsection"
                parent = "Section"
                grandparent = "Title"
                current_document.metadata["level"] = 3

            # <H2/>
            elif "Section" in current_document.metadata:
                _self = "Section"
                parent = "Title"
                grandparent = None
                current_document.metadata["level"] = 2

            # <H1/> (already set)
            elif "Title" in current_document.metadata:
                continue

            # Otherwise, error
            else:
                raise ValueError(
                    f"Metadata headers should be one of Union[Title, Section, Subsection, Detailed Subsection]. Got `{current_document.metadata}` instead.")

            # Enhance metadata
            current_document.metadata["id"] = current_document.metadata[_self]
            current_document.metadata["summary"] = None

            # update V
            self.V[current_document.metadata["id"]] = current_document

            # Add parent if not exists (i.e. parent is a no content node)
            # Note: the Langchain MD splitter ignores headings that contain no content below
            if (parent_id := current_document.metadata.get(parent)) not in self.V:
                parent_document = Document(
                    page_content="",
                    metadata={
                        # Parent must be at least <H2>...
                        "id": parent_id,
                        "summary": None,
                        "children": [],
                        "parent": current_document.metadata["Title"],
                        "Title": current_document.metadata["Title"],
                        "Section": current_document.metadata["Section"],
                    }
                )
                
                # ... and at most <H3>
                if current_document.metadata['level'] == 4:
                    parent_document.metadata["parent"] = current_document.metadata["Section"],
                    parent_document.metadata['Subsection'] = current_document.metadata['Subsection']

                # Add parent to V
                self.V[parent_id] = parent_document

                # Update grandparent
                if (grandparent_id := current_document.metadata.get(grandparent)):
                    self.V[grandparent_id].metadata['children'].append(
                        self.V[parent_id].metadata['id']
                    )

            # Update parent
            self.V[parent_id].metadata["level"] = current_document.metadata['level'] - 1
            self.V[parent_id].metadata['children'].append(
                current_document.metadata['id']
            )

    def create_summaries_recursively(self, agent: Runnable):
        def _run_recursion(curr=self.root, agent=agent):
            # Base case
            if not curr.metadata['children']:
                curr.metadata['summary'] = agent.invoke({'input': curr.page_content})
                return

            # Recursion
            for child in curr.metadata['children']:
                _run_recursion(curr=self.V[child], agent=agent)

            # Generate parent summary...
            summaries = [
                self.V[child].metadata['summary']
                for child in curr.metadata['children']
                if self.V[child].metadata.get('summary')
            ]

            curr.metadata['summary'] = agent.invoke({
                'input': StringParser.from_array([curr.page_content] + summaries)
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
