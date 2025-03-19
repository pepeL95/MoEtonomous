from moe.examples.dir2rag.experts import linear_retry

class TocNode:
    def __init__(self, level, title, page):
        """
        A single node in the TOC tree.
        :param level: Hierarchical level (1 for top-level, etc.)
        :param title: Section title.
        :param page: Starting page number (assumed 1-indexed).
        """
        self.level = level
        self.title = title
        self.page = page         # starting page (1-indexed)
        self.children = []       # list of TocNode children
        self.content = ""        # will store extracted text for this node
        self.abstract = ""       # will store abstract for this node
        self.end_page = None     # to be computed (1-indexed, inclusive)

    def __repr__(self):
        return f"TocNode(level={self.level}, title={self.title!r}, page={self.page}, end_page={self.end_page})"


class Toc:
    def __init__(self, toc_list):
        """
        Builds the TOC tree from a list of [level, title, page] entries.
        :param toc_list: A list where each element is [level, title, page].
        """
        self.root_nodes = []
        self.build_tree(toc_list)

    def build_tree(self, toc_list):
        """
        Constructs the tree using a stack-based approach.
        """
        stack = []
        for entry in toc_list:
            level, title, page = entry
            node = TocNode(level, title, page)
            # Pop until the top has a lower level than the current node.
            while stack and stack[-1].level >= node.level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                self.root_nodes.append(node)
            stack.append(node)

    def flatten(self, nodes=None):
        """
        Returns a flat list of all nodes (preorder).
        """
        if nodes is None:
            nodes = self.root_nodes
        flat = []
        for node in nodes:
            flat.append(node)
            if node.children:
                flat.extend(self.flatten(node.children))
        return flat

    def compute_boundaries(self, doc):
        """
        Computes an 'end_page' for each node.
        The rule is: a node’s end page is the start page of the next
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
        # node.end_page is 1-indexed and inclusive; for Python’s range, the stop is exclusive.
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
        for node in self.root_nodes:
            self.assign_content(doc, node)

    def print_tree(self, nodes=None, indent=0):
        """
        Recursively prints the tree structure along with the full content.
        """
        if nodes is None:
            nodes = self.root_nodes
        for node in nodes:
            print("  " * indent + f"{node.title} (Pages {node.page}-{node.end_page})")
            if node.content:
                print("  " * (indent + 1) + "Content:")
                for line in node.content.splitlines():
                    print("  " * (indent + 2) + line)
            if node.children:
                self.print_tree(node.children, indent + 1)

    def to_markdown(self, nodes=None, indent=0):
        """
        Recursively generates a markdown representation of the TOC tree using bullets.
        Each node is represented as a markdown bullet, followed by its full content (if present)
        indented as a sub-list.
        """
        if nodes is None:
            nodes = self.root_nodes
        markdown_lines = []
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
                markdown_lines.append(self.to_markdown(node.children, indent + 1))
        return "\n".join(markdown_lines)
    

    def synthesis_to_markdown(self, nodes=None, indent=0):
        """
        Recursively generates a markdown representation of the TOC tree using bullets.
        Each node is represented as a markdown bullet, followed by its full content (if present)
        indented as a sub-list.
        """
        if nodes is None:
            nodes = self.root_nodes
        markdown_lines = []
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
    
    @linear_retry(max_retries=1000, delay=5)
    def summarize_node(self, title, content, agent):
        """Invoke the section summarizer agent and return the results, with a 1 second delay"""
        payload = {
            "table_of_contents": self.to_markdown(),
            "section_title": title,
            "content": content
        }
        
        return agent.invoke({"input": payload})
    
    @linear_retry(max_retries=1000, delay=5)
    def synthesize_node(self, content, agent):
        """Invoke the section synthesizer agent and return the results, with a 1 second delay"""
        return agent.invoke({"input": content})
    
    def summarize_toc(self, agent_sigma, agent_synth):
        """
        Recursively traverses the TOC tree and applies the function sigma to every node.
        
        Parameters:
        toc (Toc): An instance of the Toc class.
        sigma (function): A function that takes a TocNode as its argument.
        """
        def traverse_nodes(nodes, agent_sigma, agent_synth):
            for node in nodes:
                node.content = self.summarize_node(node.title, node.content, agent_sigma)
                node.abstract = self.synthesize_node(node.content, agent_synth)
                if node.children:
                    traverse_nodes(node.children, agent_sigma, agent_synth)
                    abstracts = [node.abstract for node in node.children]
                    abstracts.append(node.abstract)
                    node.abstract = self.synthesize_node('\n'.join(abstracts), agent_synth)
                    
        
        traverse_nodes(self.root_nodes, agent_sigma, agent_synth)