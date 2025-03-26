import os
import re
import fitz
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List, TypedDict, Dict, Union
from dev_tools.utils.clifont import print_bold, CLIFont
from langchain_core.output_parsers import JsonOutputParser
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent

class Span(TypedDict):
    # native
    size: float
    flags: int
    font: str
    color: str
    ascender: float
    descender: float
    text: str
    origin: List[float]
    bbox: List[float]
    page: int

class LineVector(Span):
    # augmented
    wc: int # word count
    bboxh: float # log of bbox height
    bold: int # is bold
    italics: int # is italic
    start: float # probability of being the first line of a semantic block
    ignore: bool # whether to ignore this line for clustering

eob = LineVector()
eob['text'] = '<end_of_block>'
eob['size'] = 0.0
eob['flags'] = 0
eob['font'] = ''
eob['color'] = ''
eob['ascender'] = 0.0
eob['descender'] = 0.0
eob['origin'] = [-0.0, -0.0]
eob['bbox'] = [-0.0, -0.0, -0.0, -0.0]
eob['wc'] = 0
eob['bboxh'] = 0.0
eob['bold'] = 0
eob['italics'] = 0
eob['start'] = 0.0
eob['entropy'] = 0.0

sob = LineVector()
sob['text'] = '<start_of_block>'
sob['size'] = 0.0
sob['flags'] = 0
sob['font'] = ''
sob['color'] = ''
sob['ascender'] = 0.0
sob['descender'] = 0.0
sob['origin'] = [-0.0, -0.0]
sob['bbox'] = [-0.0, -0.0, -0.0, -0.0]
sob['wc'] = 0
sob['bboxh'] = 0.0
sob['bold'] = 0
sob['italics'] = 0
sob['start'] = 0.0
sob['entropy'] = 0.0

class Pdf2Markdown:
    def __init__(self, pdf_path, verbose=False):
        self.doc = fitz.open(pdf_path)
        self.features = ['entropy']
        self.verbose = verbose
        self.dfoc = self.run_etl(dropna=True)

    ########################## PUBLIC METHODS ##################################

    def get_toc(self, llm=None, pretty=False):
        if self._has_toc():
            print_bold(f"{CLIFont.light_green}Extracting ToC from PDF...{CLIFont.reset}")
            return self._xtract_toc(pretty)
        else:
            if llm is None:
                raise ValueError("llm is required when PDF has no ToC")
            print_bold(f"{CLIFont.light_green}Generating ToC from PDF...{CLIFont.reset}")
            return self._gen_toc(llm, pretty)
    
    def parse(self, pca=0):
        toc, _ = self._cluster_fonts(pca)
        pfs = self.dfoc[self.dfoc['size'] > 0 ]['size'].mode()[0]
        toc = toc[toc['size'] > pfs]
        # 1. Collect unique font sizes from the ToC.
        font_sizes = toc['size'].unique()
        # 2. Sort font sizes in descending order and assign ranks
        sizes_sorted = sorted(font_sizes, reverse=True)
        size_to_rank = {fs: i+2 for i, fs in enumerate(sizes_sorted)} # starts at <h2>
        # 3. Insert appropriate headings into the lines list based on ToC info
        for index, row in toc.iterrows():
            ln = index
            fs = row['size']
            hierarchy = "#" * size_to_rank[fs]
            self.dfoc.loc[ln, 'text'] = f"\n{hierarchy} {''.join(self.dfoc.loc[ln, 'text'].split('**'))}\n"

        # Now lines has updated heading markers where needed.
        return '\n'.join(self.dfoc['text'].to_list())
    
    def generate(self, llm, pca=0) -> str:
        toc = self._generate_toc(llm, pca=pca)
        for entry in toc:
            ln = entry["line_number"]
            level = entry['level'] + 1 # Starts at <h2>
            self.dfoc.loc[ln, 'text'] = f"\n{level * '#'} {''.join(self.dfoc.loc[ln, 'text'].split('**'))}\n"
        return '\n'.join(self.dfoc['text'].to_list())

    def run_etl(self, dropna=False):
        data = []
        for pno in range(self.doc.page_count):
            page = self._handle_page(pno)
            data.extend(page)
        df = pd.DataFrame(data)
        if dropna: df.dropna(inplace=True)
        return df
    
    ########################## PRIVATE METHODS ##################################
    
    def _has_toc(self):
        return len(self.doc.get_toc(simple=False)) > 0

    def _xtract_toc(self, pretty=False):
        toc_entries = self.doc.get_toc(simple=False)
        if pretty:
            lines = []
            for entry in toc_entries:
                level = entry[0]
                title = entry[1]
                page = entry[2]
                lines.append(f"{(level + 1) * '#'} {' '.join(title.split('\n')).strip()} - (page {page})")
            return os.linesep.join(lines)
        return toc_entries
    
    def _gen_toc(self, llm, pretty=False):
        toc_entries = self._generate_toc(llm, pca=0)
        toc = []
        for entry in toc_entries:
            level = entry['level']
            title = entry['text']
            page = entry['page']
            toc.append((level, title, page))
        
        if pretty:
            lines = []
            for entry in toc:
                level = entry[0]
                title = entry[1]
                page = entry[2] # page
                lines.append(f"{(level + 1) * '#'} {' '.join(title.split('\n')).strip()} - (page {page})")
            return os.linesep.join(lines)
        return toc
        
        
    def _generate_toc(self, llm, pca=0) -> List[Dict[str, any]]:
        toc_cluster, _ = self._cluster_fonts(pca)
        docs = [
            f"{i}: {{'text': {doc}, 'font_size': {toc_cluster.at[i, 'size']}, 'page': {int(toc_cluster.at[i, 'page'])}}}"
            for i, doc in toc_cluster['text'].items()
            if i in toc_cluster.index and doc != '</BR>'
        ]
        
        # TOC Extractor
        agent = EphemeralNLPAgent(
            name='_',
            llm=llm,
            output_parser=JsonOutputParser(),
            prompt_template=(
                "You are given a structured set of lines extracted from a PDF document. \n"
                "Each line is accompanied by its text (title) and its font_size. Your task \n"
                "is to identify and extract only the actual headings and subheadings that form \n"
                "the Table of Contents structure of this document.\n\n"
                "Criteria for identifying headings and subheadings:\n"
                f"\t* Typically, headings have larger font sizes compared to body text, though subheadings may share the same font size as standard text but follow a numeric hierarchical structure (e.g., 1, 1.1, 2.1.2).\n"
                "\t* Headings and subheadings are often bolded and begin with numeric identifiers or are concise phrases summarizing content sections.\n"
                "\t* Non-heading lines often include author names, arbitrary numerical values, random phrases, or explanatory sentences that do not represent structural components of the document’s Table of Contents.\n\n"
                "## Candidates for ToC\n"
                "-- Begin Cluster --\n"
                "{input}\n"
                "-- End Cluster --\n\n"
                "## Output Format\n"
                "Return a JSON as follows:\n"
                "- `\"line_number\"`: the original line index in the PDF extraction (an integer)\n"
                "- `\"text\"`: the text of the heading (a string).\n"
                "- \"level\": the nested hierarchy determined by you\n"
                "- \"page\": the page number of the heading/subheading"

            ),
        )

        if self.verbose: print_bold(f"{CLIFont.light_green}Sending {llm.get_num_tokens(', '.join(docs))} tokens...{CLIFont.reset}")
        ctx = '\n'.join(docs)
        # print(ctx)
        toc = agent.invoke({'input': ctx})
        return toc
    
    def _cluster_fonts(self, pca=0):  
        # Filter out zero entropy lines
        df = self.dfoc[self.dfoc['ignore'] == False]
        df = df[df['entropy'] > 0]
        
        # Scale features
        df_scaled = self._scale_features(df, self.features)

        # Kmeans
        clusters = self._cluster(data=df_scaled, pca=pca)
        df['kmeans'] = clusters
        df_scaled['kmeans'] = clusters
        golden_cluster = df[df['kmeans'] == self._which_cluster(df_scaled)]

        # Further cluster if needed...
        if len(golden_cluster) > 40:
            if self.verbose:
                print(f'{CLIFont.blue} Cluster too big, computing kmeans again...{CLIFont.reset}')
            df_scaled = self._scale_features(golden_cluster, ['entropy'])
            clusters = self._cluster(data=df_scaled, pca=0)
            golden_cluster.loc[:, 'kmeans'] = clusters
            golden_cluster = golden_cluster[golden_cluster['kmeans'] == self._which_cluster(golden_cluster)]

        return golden_cluster, df

    def _scale_features(self, data:List[dict] | pd.DataFrame, features:List[str]) -> np.ndarray:
        df = pd.DataFrame(data)
        
        # Encode non-numeric features...
        encoder = LabelEncoder()
        df['font'] = encoder.fit_transform(df['font'])
        df['flags'] = encoder.fit_transform(df['flags'])

        # Build data matrix with new encoded vals
        X = df[features]
        
        # Feature scaling...
        scaler = MinMaxScaler(feature_range=(0,1))
        # scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df = pd.DataFrame(data=X_scaled, columns=features)

        return df
    
    def _cluster(self, data, pca=0):
        # Perfom PCA for dim reduction
        df_pca = None
        if pca > 0:
            pca = PCA(n_components=pca)
            X_pca = pca.fit_transform(data)

            # Get pca data
            df_pca = pd.DataFrame(X_pca)
        
        # Kmeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        # kmeans = DBSCAN(eps=0.7, min_samples=5)
        clusters = kmeans.fit_predict(df_pca if df_pca is not None else data)
        return clusters
    
    def _compute_true_mean(self, series):
        """Compute the mean excluding outliers using the IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        filtered_series = series[(series >= (Q1 - 1.5 * IQR)) & (series <= (Q3 + 1.5 * IQR))]
        return filtered_series.mean()
    
    def _which_cluster(self, data:pd.DataFrame):
        mean_entropy = data.groupby('kmeans')['entropy'].agg(self._compute_true_mean)
        return (mean_entropy).idxmax()
    
    def _score_heading_candidate(self, string: str) -> float:
        """
        Scores how likely a string is to be a heading in a PDF document using multiple weighted heuristics.
        Returns a score between 0-1, with higher values indicating greater likelihood of being a heading.
        
        Key heuristics:
        - Length and word count patterns typical of headings
        - Capitalization patterns
        - Presence of section numbering
        - Absence of sentence-ending punctuation
        - Presence of heading-specific terms
        """
        # Initialize base score
        score = 0.0
        
        # Clean and prepare string
        clean_str = string.strip()
        tokens = clean_str.split()
        words = re.findall(r'\b[a-zA-Z]+\b', clean_str)
        token_count = len(tokens)
        word_count = len(words)
        
        if not words or len(string.strip()) < 3:
            return 0.0
        
        if word_count / token_count < 0.5:
            return 0.0

        # Length-based scoring
        if word_count <= 8:  # Ideal heading length
            score += 0.6
        elif word_count <= 10:
            score += 0.4
        else:  # Too long, likely paragraph
            for i in range(1, word_count - 11):
                score -= 0.2 * (i + 1)  # Increased penalty for very long text
        
        # Section number pattern (e.g., "1.", "1.2", "A.", "I.")
        if re.match(r'^(?:(?:\d+\.)+\d*\s|\d+\.\s|\d\s+[A-Z]|[A-Z]\.\s|[IVX]+\.\s)', clean_str):
            score += 0.9
            
        # Capitalization patterns
        if ' '.join(words).istitle():  # Title Case
            score += 0.5
        elif all([word.isupper() for word in words]):  # ALL CAPS
            score += 0.5
        elif words[0][0].isupper():  # First word capitalized
            score += 0.2
        elif not not words[0][0].isupper():  # First word *not* capitalized
            score -= 0.6
            
        # Penalize sentence-ending punctuation
        if re.search(r'[.,;:]$', clean_str):
            score -= 0.6
            
        # Penalize certain patterns...
        if re.search(r'[@#$§%*()^_+=\[\]{}<>]', clean_str):  # Special characters
            score -= 0.8
        if any(len(word) > 25 for word in words):  # Unusually long words
            score -= 0.3
        if re.search(r'\b(https?:|www\.|e-?mail)', clean_str.lower()):  # URLs/emails
            score -= 1.0

        # Clamp final score
        return max(0.0, score)
    
    def _compute_heading_entropy(self, block: List[Dict[str, Union[str, float, int]]]) -> List[Dict[str, Union[str, float, int]]]:
        def predict(title_score, weight, size, is_first):
            """Weighted sum of features."""
            if any([title_score == 0.0, is_first == 0.0]):
                return 0.0
            return round(title_score + weight * 0.5 + (size * 0.1), 2)

        # Process remaining elements (sliding window = 3)
        loi = {'</BR>', '<start_of_block>'}
        for i in range(len(block) - 2):
            curr = block[i + 1]  # vector
            prefix = block[i]['text'].strip()  # string
            suffix = block[i + 2]  # vector
            
            # Ignore breaklines
            if curr['text'].strip() in loi.union('<end_of_block>'):
                curr['entropy'] = 0.0
                continue
            
            text = ''.join(curr['text'].split('*')).strip()
            title_score = self._score_heading_candidate(text)

            # Good candidate if preceded by </BR> or <start_of_block>
            if prefix in loi:
                # Build feature vectors safely
                v0 = np.array([
                    curr.get('size', 0.0),
                    curr.get('weight', 0.0),
                ], dtype=np.float32)
                v1 = np.array([
                    suffix.get('size', 0.0),
                    suffix.get('weight', 0.0),
                ], dtype=np.float32)
                
                # Dot prod similarity
                norm2 = round(np.linalg.norm(v0) ** 2, 2)
                dot = round(float(np.dot(v0, v1)), 2)
                sim = dot / norm2

                # If highly similar, treat current line as paragraph text 
                # Note: We exclude fully consecutive bolded lines, since those are likely headings
                if sim == 1.0 and not suffix.get('bold', 0.0):
                    curr['start'] = curr.get('start', 0.0)
                # Otherwise, mark it as a start (e.g., heading)
                else:
                    curr['start'] = max(curr.get('start', 0.0), 1.0)
            else:
                # Not preceded by <start_of_block>
                curr['start'] = curr.get('start', 0.0)

            # Final heading entropy score
            curr['entropy'] = predict(
                title_score,
                curr.get('weight', 0.0),
                curr.get('size', 0.0),
                curr.get('start', 0.0)
            )

        return block
    
    def _isnum(self, s:str):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _handle_page(self, pno:int) -> List[LineVector]:
        '''
        wraps page with <start_of_page>...handle_blocks...</end_of_page> tags for llm understanding
        '''
        blocks = [self._merge_lines(block) for block in self.doc.load_page(pno).get_textpage().extractDICT().get('blocks', [])]
        mask_bboxes = [table.bbox for table in self.doc.load_page(pno).find_tables()]
        line_vectors = []
        for block in blocks:
            block_ = self._handle_block(block, pno, mask_bboxes)
            line_vectors.extend(block_)
        
        return line_vectors

    def _handle_block(self, block, pno, mask_bboxes) -> List[LineVector]:
        '''
        wraps block with <start_of_block>...handle_lines...</end_of_block> tags for llm understanding
        '''
        lines = []

        # Process lines
        lines.append(sob)
        for line in block.get('lines', []):
            lines.append(self._handle_line(line, pno, mask_bboxes))
        lines.append(eob)

        # Predict first lines of a true text block
        lines = self._compute_heading_entropy(lines)
        return lines

    def _handle_line(self, line:List[Span], pno:int, mask_bboxes) -> LineVector:
        def is_bold(span):
            if span['text'].strip() in {'</BR>', '!', ''}: # Note '!' is a special character for unknown ascii codes
                return True
            return any(bold_match in span['font'] for bold_match in ["Bold", "TB", "Medi", "CMB"])
        
        def is_italic(span):
            if span['text'].strip() in {'</BR>', '!'}: # Note '!' is a special character for unknown ascii codes
                return True
            return any(bold_match in span['font'] for bold_match in ["oblique", "CMTI", "CMMI", "Ital"]) 

        
        # Handle breaklines
        if line and not line[0]['text'].strip():
            line[0]['text'] = '</BR>'
        
        # Reduce...
        line_vector = LineVector()
        line_vector['text'] = ''.join(span['text'] for span in line).strip()
        line_vector['bold'] = 1.0 * all(is_bold(span) for span in line)
        line_vector['italics'] = 1.0 * all(is_italic(span) for span in line)
        line_vector['font'] = line[0]['font']
        line_vector['color'] = line[0]['color']
        line_vector['size'] = line[0]['size']
        line_vector['flags'] = line[0]['flags']
        line_vector['ascender'] = line[0]['ascender']
        line_vector['descender'] = line[0]['descender']
        line_vector['origin'] = line[0]['origin']
        line_vector['bbox'] = line[0]['bbox']
        line_vector['page'] = pno

        # Augment...
        ret = self._augment_line_vector(line_vector, mask_bboxes)
        return ret
            
    def _augment_line_vector(self, line_vector:LineVector, mask_bboxes):
        line_vector['entropy'] = self._score_heading_candidate(''.join(line_vector['text'].split('*')).strip())
        line_vector['wc'] = len(line_vector['text'].split())
        line_vector['start'] = 0.0
        line_vector['weight'] = line_vector['bold'] + line_vector['italics']
        line_vector['bboxh'] = np.log(abs(line_vector['bbox'][1] - line_vector['bbox'][3]))
        line_vector['ignore'] = self._ignore_bboxes(line_vector, mask_bboxes)
        if line_vector['text'][0] == '#':
            line_vector['text'] = f"`{line_vector['text']}`"
        if line_vector['bold'] and line_vector['text'] != '</BR>':
            line_vector['text'] = f"**{line_vector['text']}**"
        if line_vector['italics'] and line_vector['text'] != '</BR>':
            line_vector['text'] = f"*{line_vector['text']}*"
            line_vector['size'] -= 0.01
        return line_vector
    
    def _merge_lines(self, block:List) -> List:
        '''
        Merges lines by y0 (..if on the same column)
        '''
        true_lines = {}
        for line in block.get('lines', []):
            y0, y1 = int(line['bbox'][1]), int(line['bbox'][3])

            # Mapping out new lines by their y0, in order to merge lines (+/-1)
            if not y0 in true_lines:
                true_lines[y0] = line.get('spans', [])
            
            # The following lines should be merged (unless document is column-designed)
            else:
                if len(line.get('spans', [])):
                    # prev word and next word must be in the same half of the page
                    doc_with = self.doc[0].bound()[2] - self.doc[0].bound()[0]
                    if (
                        true_lines[y0][-1]['bbox'][2] > doc_with / 2 and line['spans'][0]['bbox'][0] > doc_with / 2 or
                        true_lines[y0][-1]['bbox'][2] < doc_with / 2 and line['spans'][0]['bbox'][0] < doc_with / 2
                    ):
                        line['spans'][0]['text'] = f" {line['spans'][0]['text']}"
                        true_lines[y0].extend(line['spans'])
                   
        return {'lines': list(true_lines.values())}

    def _ignore_bboxes(self, line_vector:LineVector, bboxes) -> bool:
        x0, y0 = line_vector['origin']
        # Headers
        if y0 < 50:
            return True
        # Footers
        if y0 > self.doc[0].bound()[3] - 50:
            return True
        # Provided bboxes
        for bbox in bboxes:
            if x0 > bbox[0] and x0 < bbox[2] and y0 > bbox[1] and y0 < bbox[3]:
                return True
        return False