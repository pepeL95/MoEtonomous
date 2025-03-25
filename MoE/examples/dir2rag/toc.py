import os
from langchain_core.documents import Document
import tiktoken
import fitz
import re



class Tr3OCNode:
    def __init__(self, level, title, page, src, document_title):
        """
        A single node in the TOC tree.
        :param level: Hierarchical level (1 for top-level, etc.)
        :param title: Section title.
        :param page: Starting page number (assumed 1-indexed).
        """
        self.level = level
        self.title = title
        self.page = page         # starting page (1-indexed)
        self.src = src          # source file name
        self.document_title = document_title
        self.children = []       # list of TocNode children
        self.content = ""        # will store extracted text for this node
        self.abstract = ""       # will store abstract for this node
        self.summary = ""        # will store summary for this node
        self.parent = ""       # will store parent for this node
        self.end_page = ""     # to be computed (1-indexed, inclusive)
        self.node_id = ""
        self.parent_id = ""

    def __repr__(self):
        return f"TocNode(level={self.level}, title={self.title!r}, page={self.page}, end_page={self.end_page})"

    def count_node_tokens(self, model="gpt-4"):
        """
        Count the number of tokens in the given text using the specified model's tokenizer.
        If no text is provided, counts tokens in the node's content.

        Args:
            text (str, optional): Text to count tokens for. If None, uses node's content.
            model (str): The model to use for tokenization. Defaults to "gpt-4".

        Returns:
            int: Number of tokens in the text
        """
        return Tr3OCNode.count_tokens(self.content)

    @staticmethod
    def count_tokens(text, model="gpt-4"):
        """
        Count the number of tokens in the given text using the specified model's tokenizer.
        If no text is provided, counts tokens in the node's content.

        Args:
            text (str, optional): Text to count tokens for. If None, uses node's content.
            model (str): The model to use for tokenization. Defaults to "gpt-4"""

        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return len(text)

    def to_dict(self):
        """
        Converts the Tr3OCNode instance to a dictionary representation.

        Returns:
            dict: A dictionary containing the node's properties
        """
        return {
            "level": self.level,
            "title": self.title,
            "page": self.page,
            "end_page": self.end_page,
            "content": self.content,
            "abstract": self.abstract,
            "summary": self.summary,
            "src": self.src,
            "document_title": self.document_title,
            "node_id": self.node_id,
            "parent_id": self.parent_id,
        }

    def from_dict(self, dict):
        """
        Converts a dictionary representation back to a Tr3OCNode instance.

        Args:
            dict: A dictionary containing the node's properties
        """
        ret = Tr3OCNode(
            level=dict["level"],
            title=dict["title"],
            page=dict["page"],
            src=dict["src"],
            document_title=dict["document_title"]
        )
        ret.content = dict["content"]
        ret.abstract = dict["abstract"]
        ret.summary = dict["summary"]
        ret.end_page = dict["end_page"]
        ret.node_id = dict["node_id"]
        ret.parent_id = dict["parent_id"]

        return ret



class Tr3OC:
    """
    A class for representing a TOC tree.
    """
    def __init__(self, title, src):
        self.title = title
        self.src = src
        self.references = ""
        self.summary = ""
        self.abstract = ""
        self.root = Tr3OCNode(0, title, 0, src, title)

    def build_from_toc_list(self, toc_list):
        self.from_toc_list(toc_list)

    def from_toc_list(self, toc_list):
        """
        Builds the TOC tree from a list of [level, title, page] entries.
        :param toc_list: A list where each element is [level, title, page].
        """
        self.references = ""
        self.root = Tr3OCNode(0, self.title, 0, self.src, self.title)
        self.build_tree_from_doc_toc(toc_list)
        if self.root.children:
            self.root.end_page = self.root.children[0].page if self.root.children[0].page != 1 else 2  # Root should have some content
        return self

    @staticmethod
    def from_dict(dict_list):
        """
        Constructs a Tr3OC instance from a dictionary representation.

        Args:
            dict_list (list): A list of dictionaries representing the TOC structure,
                            as produced by tree_to_dict()

        Returns:
            Tr3OC: A new Tr3OC instance with the structure from the dictionary
        """
        toc = Tr3OC([])  # Create empty instance
        toc.root_nodes = Tr3OC._nodes_from_dict(dict_list)
        return toc

    @staticmethod
    def _nodes_from_dict(dict_list):
        """
        Helper method to recursively construct Tr3OCNode instances from dictionary data.

        Args:
            dict_list (list): List of dictionaries representing nodes

        Returns:
            list: List of constructed Tr3OCNode instances
        """
        nodes = []
        for node_dict in dict_list:
            # Create new node with basic properties
            node = Tr3OCNode.from_dict(node_dict)

            # Recursively process children if they exist
            if "children" in node_dict:
                node.children = Tr3OC._nodes_from_dict(node_dict["children"])
                # Set parent references
                for child in node.children:
                    child.parent = node

            nodes.append(node)
        return nodes

    def search_by_title(self, title, node='root'):
        """
        Recursively searches for a node with the given title in this node's subtree.

        Args:
            title (str): The title to search for

        Returns:
            Tr3OCNode or None: The first node with matching title, or None if not found
        """
        if node == "root":
            node = self.root

        if node.title == title:
            return node

        for child in node.children:
            result = self.search_by_title(title, child)
            if result:
                return result

        return None

    def tree_to_dict(self, nodes=None):
        """
        Converts the TOC tree into a dictionary representation.
        Each node is represented as a dictionary with its properties and children.

        Returns:
            list: A list of dictionaries representing the root nodes and their children
        """
        if nodes is None:
            nodes = self.root.children

        result = []
        for node in nodes:
            node_dict = node.to_dict()
            if node.children:
                node_dict["children"] = self.tree_to_dict(node.children)

            result.append(node_dict)

        return result

    def build_tree_from_doc_toc(self, toc_list):
        """
        Constructs the tree using a stack-based approach.
        """
        stack = []
        for entry in toc_list:
            level, title, page = entry
            node = Tr3OCNode(level, title, page, self.src, self.title)
            # Pop until the top has a lower level than the current node.
            while stack and stack[-1].level >= node.level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                node.parent = self.root
                self.root.children.append(node)
            stack.append(node)

    def flatten(self, nodes=None):
        """
        Returns a flat list of all nodes (preorder).
        """
        if nodes is None:
            nodes = self.root.children
        flat = []
        for node in nodes:
            flat.append(node)
            if node.children:
                flat.extend(self.flatten(node.children))
        return flat

    def _assign_node_ids(self):
        """
        Assigns a unique node_id to every node in this tree, including the root.
        We'll track parent_id as well for convenience.
        """
        current_id = 0
        self.root.node_id = current_id
        self.root.parent_id = ""
        queue = [self.root]
        while queue:
            parent = queue.pop(0)
            for child in parent.children:
                current_id += 1
                child.node_id = current_id
                child.parent_id = parent.node_id
                queue.append(child)

    def _flatten_with_root(self):
        """
        Like flatten(), but includes the root node itself as the first item.
        """
        def _collect(node):
            nodes = [node]
            for c in node.children:
                nodes.extend(_collect(c))
            return nodes
        return _collect(self.root)

    def compute_boundaries(self, doc):
        """
        Computes an 'end_page' for each node.
        The rule is: a node's end page is the start page of the next
        TOC entry (in preorder) that is at the same or a higher level.
        If none exists, we assume the node extends to the end of the document.
        """
        flat_nodes = self.flatten()
        total_pages = doc.page_count  # Total number of pages (doc.load_page uses 0-index)
        for i, node in enumerate(flat_nodes):
            # Default: extend to the end of the document.
            end_page = total_pages
            # Look for the next node in the flat list with level <= current node.
            for next_node in flat_nodes[i+1:]:
                if next_node.level <= node.level:
                    # Set the end page to the next node's start page (1-indexed)
                    end_page = next_node.page
                    break
            node.end_page = end_page

    def assign_content(self, doc, node):
        """
        Recursively extracts and assigns text to each node.
        If a node has children, we assume:
          - The parent's "intro" is the text from its start page up to the page before its first child's start.
          - Any content after the last child (if present) is taken as a "post-child" segment.
        For leaf nodes, we simply extract text from start to end.
        """
        # Convert starting page to 0-indexed
        start_idx = node.page - 1
        # node.end_page is 1-indexed and inclusive; for Python's range, the stop is exclusive.
        end_idx = node.end_page

        if node.children:
            # --- Parent's introduction ---
            first_child_start_idx = node.children[0].page - 1
            intro_text = ""
            if start_idx < first_child_start_idx:
                for p in range(start_idx, first_child_start_idx):
                    intro_text += doc.load_page(p).get_text()
            # --- Parent's post-child content ---
            last_child = node.children[-1]
            post_text = ""
            # If the parent's section extends beyond the last child's section,
            # include that remainder.
            if last_child.end_page < node.end_page:
                # last_child.end_page is 1-indexed; parent's post starts at next page.
                post_start_idx = last_child.end_page  # (since page numbers: last child's pages end at last_child.end_page)
                for p in range(post_start_idx, end_idx):
                    post_text += doc.load_page(p).get_text()
            # Combine and clean up (you might want to store them separately in a more complex system)
            node.content = (intro_text + "\n" + post_text).strip()
        else:
            # For leaf nodes, extract all text from start to end.
            text = ""
            for p in range(start_idx, end_idx):
                text += doc.load_page(p).get_text()
            node.content = text.strip()
        # Recurse for children.
        for child in node.children:
            self.assign_content(doc, child)

    def enhance_with_content(self, doc):
        """
        Computes boundaries and extracts content from the document,
        enhancing each TOC node with its corresponding text.
        """
        self.compute_boundaries(doc)
        for node in self.root.children:
            self.assign_content(doc, node)

    def print_tree(self, nodes=None, indent=0):
        """
        Recursively prints the tree structure along with the full content.
        """
        if nodes is None:
            nodes = self.root.children
        for node in nodes:
            print("  " * indent + f"{node.title} (Pages {node.page}-{node.end_page})")
            if node.content:
                print("  " * (indent + 1) + "Content:")
                for line in node.content.splitlines():
                    print("  " * (indent + 2) + line)
            if node.children:
                self.print_tree(node.children, indent + 1)

    def toc_to_markdown(self, nodes=None, indent=0):
        """
        Recursively generates a markdown representation of the TOC tree using bullets.
        Each node is represented as a markdown bullet, followed by title, and page range. No content is included.
        """
        markdown_lines = []

        if nodes is None:
            nodes = self.root.children
            markdown_lines.append(f"{self.root.title}")

        for node in nodes:
            markdown_lines.append("  " * indent + "- " + f"{node.title} | Page ({node.page})")

            if node.children:
                markdown_lines.append(self.toc_to_markdown(node.children, indent + 1))

        return "\n".join(markdown_lines)

    def content_to_markdown(self, nodes=None, indent=0):
        """
        Recursively generates a markdown representation of the TOC tree using bullets.
        Each node is represented as a markdown bullet, followed by its full content (if present)
        indented as a sub-list.
        """
        markdown_lines = []

        if nodes is None:
            nodes = self.root.children
            markdown_lines.append(f"{self.root.title}")

        for node in nodes:
            # Add the node header with a markdown bullet
            markdown_lines.append("  " * indent + "- " + f"{node.title} (Pages {node.page}-{node.end_page})")
            # If there is content, add a "Content:" label and then each content line indented further
            if node.content:
                markdown_lines.append("  " * (indent + 1) + "Content:")
                for line in node.content.splitlines():
                    markdown_lines.append("  " * (indent + 2) + line)
            # Recursively process any child nodes
            if node.children:
                markdown_lines.append(self.content_to_markdown(node.children, indent + 1))
        return "\n".join(markdown_lines)

    def summary_to_markdown(self, nodes=None, indent=0):
        """
        Recursively generates a markdown representation of the TOC tree using bullets.
        Each node is represented as a markdown bullet, followed by its full content (if present)
        indented as a sub-list.
        """
        markdown_lines = []

        if nodes is None:
            nodes = self.root.children
            markdown_lines.append(f"{self.root.title}")

        for node in nodes:
            # Add the node header with a markdown bullet
            markdown_lines.append("  " * indent + "- " + f"{node.title} (Pages {node.page}-{node.end_page})")
            # If there is content, add a "Content:" label and then each content line indented further
            if node.summary:
                markdown_lines.append("  " * (indent + 1) + "Content:")
                for line in node.summary.splitlines():
                    markdown_lines.append("  " * (indent + 2) + line)
            # Recursively process any child nodes
            if node.children:
                markdown_lines.append(self.summary_to_markdown(node.children, indent + 1))
        return "\n".join(markdown_lines)


    def synthesis_to_markdown(self, nodes=None, indent=0):
        """
        Recursively generates a markdown representation of the TOC tree using bullets.
        Each node is represented as a markdown bullet, followed by its full content (if present)
        indented as a sub-list.
        """
        markdown_lines = []

        if nodes is None:
            nodes = self.root.children
            markdown_lines.append(f"{self.root.title}")

        for node in nodes:
            # Add the node header with a markdown bullet
            markdown_lines.append("  " * indent + "- " + f"{node.title} (Pages {node.page}-{node.end_page})")
            # If there is content, add a "Content:" label and then each content line indented further
            if node.abstract:
                markdown_lines.append("  " * (indent + 1) + "Content:")
                for line in node.abstract.splitlines():
                    markdown_lines.append("  " * (indent + 2) + line)
            # Recursively process any child nodes
            if node.children:
                markdown_lines.append(self.synthesis_to_markdown(node.children, indent + 1))
        return "\n".join(markdown_lines)

    def summarize_node(self, title, content, agent):
        """Invoke the section summarizer agent and return the results, with a 1 second delay"""
        payload = {
            "table_of_contents": self.toc_to_markdown(),
            "section_title": title,
            "content": content
        }

        tokens = Tr3OCNode.count_tokens(content)
        
        if tokens == 0:
            return ""
        
        print(f"Summarizing node {title} | tokens: {tokens}")
        return agent.invoke({"input": payload}, tokens)

    def synthesize_node(self, title, content, agent):
        """Invoke the section synthesizer agent and return the results, with a 1 second delay"""
        tokens = Tr3OCNode.count_tokens(content)
        
        if tokens == 0:
            return ""
        
        print(f"Synthesizing node {title} | tokens: {tokens}")
        return agent.invoke({"input": content}, tokens)

    def extract_metadata(self, doc: fitz.Document, agent):
        """Invoke the document metadata extractor agent and return the results, with a 1 second delay"""
        metadata = {
            "title": None,
            "authors": None,
            "date": None
        }

        for pno in range(doc.page_count):
            # If metadata all the metadatas are complete, break
            if all(metadata.values()):
                break
            # Extract metadata from the page
            content = doc.load_page(pno).get_text()
            tokens = Tr3OCNode.count_tokens(content)
            _metadata = agent.invoke({"input": content}, tokens)
            # Update metadata with the new non-None values
            metadata = {k: v for k, v in _metadata.items() if k in metadata and v}

        return metadata

    def traverse(self, node, fn, results={}):
        """
        Recursively process nodes, applying fn to each node.
        """
        try:
            results[node.title] = fn(node)
        except Exception as e:
            results[node.title] = str(e)

        for child in node.children:
            self.traverse(child, fn, results)

    def summarize_toc(self, agent_sigma):
        """
        Recursively traverses the TOC tree and applies summarization to each node.

        Parameters:
            agent_sigma: Agent for summarizing individual sections
        """
        def _summarize_nodes(nodes, agent_sigma):
            """
            Recursively process nodes, generating summaries.
            """
            for i, node in enumerate(nodes):
                try:
                    print(f"Summarizing node {node.title}")
                    node.summary = self.summarize_node(node.title, node.content, agent_sigma)
                    if node.children:
                        _summarize_nodes(node.children, agent_sigma)
                except Exception as e:
                    print(f"Error Summarizing node {node.title}: {e}")
                    break

        _summarize_nodes(self.root.children, agent_sigma)

    def synthesize_toc(self, agent_synth):
        """
        Recursively traverses the TOC tree and applies synthesis to each node.

        Parameters:
            agent_synth: Agent for synthesizing summaries into abstracts
        """
        def _synth_nodes(nodes, agent_synth):
            """
            Recursively process nodes, generating abstracts.
            Returns list of abstracts from processed nodes.
            """
            abstracts = []
            for i, node in enumerate(nodes):
                try:
                    print(f"Synthesizing node {node.title}")
                    if node.children:
                        children_abstracts = _synth_nodes(node.children, agent_synth)
                        combined_content = '\n\n'.join([node.summary] + children_abstracts)
                        node.abstract = self.synthesize_node(node.title, combined_content, agent_synth)
                    else:
                        node.abstract = self.synthesize_node(node.title, node.summary, agent_synth)
                    abstracts.append(node.abstract)
                except Exception as e:
                    print(f"Error processing node {node.title}: {e}")
                    break
            return abstracts

        _synth_nodes(self.root.children, agent_synth)

        doc_abstract_lines = os.linesep.join([node.abstract for node in self.root.children])
        tokens = Tr3OCNode.count_tokens(doc_abstract_lines)
        print(f"Synthesizing document {self.title} | tokens: {tokens}")
        self.abstract = agent_synth.invoke({"input": doc_abstract_lines}, tokens)


    @staticmethod
    def join_text_sections(text1, text2):
        """Joins two text sections by consolidating any overlapping text.

        If one text is contained in the other, returns the longer text.
        Otherwise, finds the longest suffix of text1 that matches a prefix of text2 and
        merges them without duplicating the overlapping portion.
        """
        if not text1:
            return text2
        if text1 in text2:
            return text2
        if text2 in text1:
            return text1
        overlap_length = 0
        max_possible = min(len(text1), len(text2))
        for i in range(max_possible, 0, -1):
            if text1.endswith(text2[:i]):
                overlap_length = i
                break
        return text1 + text2[overlap_length:]

    def document_to_markdown(self):
        """Generates a markdown representation of the entire document by merging the content
        of all TOC nodes, consolidating overlapping text segments.

        This method flattens the TOC tree, then iteratively joins each node's content using
        the join_text_sections helper to remove duplicate overlapping parts.
        The final output starts with the document's title as a header.
        """
        flat_nodes = self.flatten()
        merged_content = ""
        for node in flat_nodes:
            if node.content.strip():
                if merged_content:
                    merged_content = Tr3OC.join_text_sections(merged_content, node.content)
                else:
                    merged_content = node.content
        return f"# {self.root.title}\n\n{merged_content}"

    @staticmethod
    def extract_section_by_boundaries(text, start_marker, end_marker=None):
        """
        Extracts a section from the provided text based on start and optional end boundaries.

        Parameters:
            text (str): The full text of the document.
            start_marker (str): The exact text that marks the start of the section (case-insensitive).
            end_marker (str, optional): The exact text that marks the end of the section (case-insensitive). If None, extracts to the end of the document.

        Returns:
            str: The extracted section, or a message if start marker is not found.
        """
        lines = text.split('\n')
        start_idx = ""
        end_idx = ""

        # Locate start marker
        for idx, line in enumerate(lines):
            if line.strip().lower() == start_marker.strip().lower():
                start_idx = idx
                break

        if not start_idx:
            return "Start marker not found in the text."

        # Locate end marker if provided
        if end_marker:
            for idx in range(start_idx + 1, len(lines)):
                if line.strip().lower() == end_marker.strip().lower():
                    end_idx = idx
                    break

        if not end_idx:
            end_idx = len(lines)

        extracted_section = "\n".join(lines[start_idx:end_idx])

        return extracted_section

    def to_documents(self):
        """
        Serializes this entire Tr3OC (including the root) into a list of Documents.
        Each node is converted into a Document whose metadata includes hierarchy references
        (node_id, parent_id) so we can reconstruct the tree.
        """
        # 1) Assign unique IDs to every node in the tree
        self._assign_node_ids()

        # 2) Flatten the tree *including* the root node
        all_nodes = self._flatten_with_root()

        documents = []
        for node in all_nodes:
            metadata = node.to_dict()
            metadata["content"] = ""
            metadata["document_abstract"] = self.abstract
            metadata["document_summary"] = self.summary
            
            doc = Document(
                page_content=node.content,
                metadata=metadata
            )
            documents.append(doc)
        
        if self.references:
            ref_metadata = self.references.to_dict()
            ref_metadata["content"] = ""
            ref_metadata["document_abstract"] = self.abstract
            ref_metadata["document_summary"] = self.summary
            
            ref_doc = Document(
                page_content=self.references.content,
                metadata=ref_metadata
            )
            
            documents.append(ref_doc)    
        
        return documents


    @classmethod
    def build_from_documents(cls, documents):
        """
        Reconstructs a Tr3OC (including its root) from a list of Documents whose metadata
        includes hierarchy references (node_id, parent_id). The node with parent_id=None
        is considered the root.
        """
        if not documents:
            # Return an empty, generic Tr3OC
            return cls(title="Empty", src="Unknown")

        # 1) Identify which Document is the root (parent_id=None)
        root_docs = [d for d in documents if not d.metadata.get("parent_id")]
        if not root_docs:
            # If we find no explicit root doc, default to the first doc in the list
            root_doc = documents[0]
        else:
            root_doc = root_docs[0]

        # 2) Create the new Tr3OC instance with the root’s info
        title = root_doc.metadata.get("title", "Root")
        src = root_doc.metadata.get("src", "generated")
        instance = cls(title=title, src=src)

        instance.root.level = root_doc.metadata.get("level", 0)
        instance.root.title = root_doc.metadata.get("title", "Root")
        instance.root.page = root_doc.metadata.get("page", 0)
        instance.root.end_page = root_doc.metadata.get("end_page")
        instance.root.document_title = root_doc.metadata.get("document_title", title)
        instance.root.abstract = root_doc.metadata.get("abstract", "")
        instance.root.summary = root_doc.metadata.get("summary", "")
        instance.root.content = root_doc.page_content

        # Mark the root's node_id and parent_id
        instance.root.node_id = root_doc.metadata.get("node_id", 0)
        instance.root.parent_id = ""

        # 3) Dictionary to hold all nodes by their node_id
        all_nodes = {instance.root.node_id: instance.root}

        # 4) Create Tr3OCNode objects for other docs, then link them up
        for doc in documents:
            if doc is root_doc:
                continue
            
            node_id = doc.metadata.get("node_id")
            parent_id = doc.metadata.get("parent_id")
            level = doc.metadata.get("level", 1)
            title = doc.metadata.get("title", "Node")
            page = doc.metadata.get("page", 1)
            end_page = doc.metadata.get("end_page")
            src = doc.metadata.get("src", "generated")
            document_title = doc.metadata.get("document_title", instance.root.title)

            node = Tr3OCNode(level, title, page, src, document_title)
            node.end_page = end_page
            node.content = doc.page_content or ""
            node.abstract = doc.metadata.get("abstract", "")
            node.summary = doc.metadata.get("summary", "")
            node.node_id = node_id
            node.parent_id = parent_id

            all_nodes[node_id] = node

        # 5) Link each node to its parent
        for node_id, node in all_nodes.items():
            if node_id == instance.root.node_id:
                continue
            
            parent_id = node.parent_id
            
            if parent_id in all_nodes:
                parent_node = all_nodes[parent_id]
                node.parent = parent_node
                parent_node.children.append(node)

        return instance

