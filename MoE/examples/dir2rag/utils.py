import os
import arxiv
import pandas as pd

class arx:
    """Arxiv Util"""

    @staticmethod
    def index_recent(query="cat:cs.*", max_results=5):
        """Download search index as dataframe"""        
        client = arxiv.Client()

        # Create a search query for Computer Science, sorted by the latest submissions
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # Fetch and return the results using the client
        data = []
        try:
            for result in client.results(search):
                data.append({
                    'id': os.path.basename(result.entry_id),
                    'title': result.title,
                    'pdf_url': result.pdf_url,
                    'authors': ", ".join([author.name for author in result.authors]),
                    'published': result.published,
                    'updated': result.updated,
                    'summary': result.summary,
                    'primary_category': result.primary_category,
                    'related_categories': result.categories
                })
        except Exception as exc:
            print(f"Completed with errors: {exc}")
        
        # Convert the list of dictionaries to a DataFrame
        publications_df = pd.DataFrame(data)
        return publications_df

    @staticmethod
    def fetch_by_id(id):
        """Download document index, by id"""
        client = arxiv.Client()
        search = arxiv.Search(id_list=[id])
        result = next(client.results(search))

        ret = {
            'title': result.title,
            'pdf_url': result.pdf_url,
            'authors': ", ".join([author.name for author in result.authors]),
            'published': result.published,
            'updated': result.updated,
            'summary': result.summary,
            'primary_category': result.primary_category,
            'related_categories': result.categories
        }
        
        return ret

    @staticmethod
    def download_file(doc_id, dirpath, filename, format='pdf'):
        """Download pdf document, by id"""
        client = arxiv.Client()
        paper = next(client.results(arxiv.Search(id_list=[doc_id])))
        
        if format == 'src':
            paper.download_source(dirpath=dirpath, filename=filename)
        else:
            paper.download_pdf(dirpath=dirpath, filename=filename)
        
    
    @staticmethod
    def print_index_report(df, limit=None):
        print(arx.get_md(df, limit))
    
    @staticmethod
    def get_md(df, limit=None):
        count = 0
        ret = ''
        for row in df.itertuples():
            if limit is not None and count == limit:
                break
            
            ret += (
                f"### {row.title}\n"
                f"**{row.id}**</BR>\n"
                f"**{row.pdf_url}**</BR>\n"
                # f"**Description**</BR>\n- {row.abstract_highlights}"
                f"#### Highlights\n"
                f"{row.hilights}\n"
                f"<details>\n"
                f"<summary>Abstract</summary>\n"
                f"{row.summary}\n"
                f"</details>\n\n"
                f"{50 * '-'}\n\n"
            )

            count += 1
        
        return ret