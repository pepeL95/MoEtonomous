class Prompters:
    class Arxiv:
        @staticmethod
        def ApiQueryBuildFewShot():
            ret = """\
Given the following user instruction for fetching an arxiv document, extract the search query, arxiv-compatible category taxonomy, and number of desired documents.

## Output Format
Output a json in the following format.

{{
"query": this is the input search query...if nothing specific, default to an empty string, 
"cat": this is the category taxonomy (convert it to the compatible category taxonomy for the arxiv api)...if none given, default to "cs.CL",
"N": this is the number of articles requested (if none given, default to 10)
}}

## Examples
User instruction: What's new?
Answer: {{"query": "", "cat": "cs.CL", "N": 10}}

User instruction: Search for articles about graph rag. Give me 3 documents.
Answer: {{"query": "Graph%20Rag", "cat": "cs.CL", "N": 3}}

User instruction: Search for articles about multimodal prompting techniques. Give me 3 documents.
Answer: {{"query": "Multimodal%20prompting%20techniques", "cat": "cs.CL", "N": 3}}

User instruction: Fetch some documents in the astro physics category. No more than 5 papers please
Answer: {{"query": "", "cat": "astro-ph.SR", "N": 5}}

User instruction: What's recent?
Answer: {{"query": "", "cat": "cs.CL", "N": 10}}

User instruction: What's new in math?
Answer: {{"query": "", "cat": "math.", "N": 10}}

User instruction: {input}
Answer: 
"""
            return ret

        @staticmethod
        def AbstractSigma():
            ret = """\
    Summarize the abstract of the given article so that you capture the key ideas. Do not display ANY code!!!

    **Article Description**

    The article is provided in a JSON format as follows:
    {{'title': Article title, 'published_date': published date, 'pdf_link': link.to.pdf.file, 'abstract': The article's abstract}}

    **Output Format Instructions**

    ## <The article title here>
    Published date: This contains the date that the article was published.
    Text pdf link: [arxiv](<This contains the link to the pdf file for the full article in DateTime format.>)
    **Abstract Summary:** This contains the summary of the abstract (1-3 bullet points. Use "*" for bullet points.)

    **Article**

    {input}
    """
            return ret