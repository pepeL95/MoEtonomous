import os
import re
import fitz
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from typing import List, Tuple, TypedDict, Dict, Union
from dev_tools.utils.clifont import print_bold, CLIFont

class Span(TypedDict):
    # canonical span
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
    # augmented span
    wc: int  # word count
    bboxh: float  # log of bbox height
    bold: int  # is bold
    italics: int  # is italic
    start: float  # probability of being the first line of a semantic block
    ignore: bool  # whether to ignore this line for clustering
    whitespace: int | None

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
eob['whitespace'] = 0.0

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
sob['whitespace'] = 0.0

class Pdf2Markdown:
    def __init__(self, pdf_path, verbose=False):
        self.doc = fitz.open(pdf_path)
        self.features = ['size', 'entropy', 'weight', 'wc', 'flags', 'caps_ratio', 'dist2par', 'whitespace', 'page', 'text']
        self.verbose = verbose
        self.dfoc = self.run_etl(dropna=True)
        self.model = None

    ######################################################## PUBLIC METHODS ######################################################

    def invoke(self):
        if self._has_toc():
            print_bold(f"{CLIFont.light_green}Extracting ToC from PDF...{CLIFont.reset}")
            return self.parse()
        else:
            print_bold(f"{CLIFont.light_green}Generating ToC from PDF...{CLIFont.reset}")
            raise NotImplementedError('Gotta figure out a way to get the level hierarchy.')

    def get_toc(self, llm=None, pretty=False):
        if self._has_toc():
            print_bold(f"{CLIFont.light_green}Extracting ToC from PDF...{CLIFont.reset}")
            return self._xtract_toc(pretty)
        else:
            if llm is None:
                raise ValueError("llm is required when PDF has no ToC.")
            print_bold(f"{CLIFont.light_green}Generating ToC from PDF...{CLIFont.reset}")
            raise NotImplementedError('Gotta figure out a way to get the level hierarchy.')

    def parse(self):
        toc_entries = self._xtract_toc(pretty=False)
        ham = self._predict()
        matches = self._fuzzy_match(ham, toc_entries)
        # Replace the text of the lines with the matched text
        for _, row in matches.iterrows():
            self.dfoc.at[row['index'], 'text'] = row['text']

        return '\n'.join(self.dfoc['text'].to_list())

    def run_etl(self, dropna=False):
        data = []
        for pno in range(self.doc.page_count):
            page = self._handle_page(pno)
            data.extend(page)
        df = pd.DataFrame(data)
        # Apply additional augmentations
        df = self._augment_dfoc(df)
        if dropna:
            df.dropna(inplace=True)
        return df

    ######################################################## PRIVATE METHODS ######################################################

    def _fuzzy_match(self, df, toc, min_score=70):
        '''
        Inputs:
        - df: The internal df which contains all the goodies
        - toc: The true, extracted toc (if exists) from the document metadata.

        Output:
        - A pd.dataframe with the expected ToC

        High-level description:
        Given a two sets: {candidate toc elements} and {true toc elements}, fuzzy match corresponding elements. 
        Note: We use fuzzy match because they may not be syntactically equivalent.
        '''
        matches = []
        for entry in toc:
            level = entry[0]
            toc_text = entry[1].upper()
            toc_page = entry[2]
            # Get candidate lines within page margin
            candidates = df[df['page'] == (toc_page - 1)]

            # Prepare a list of lines to match against
            candidate_texts = candidates['text'].apply(
                lambda txt: ''.join(txt.split('*')).upper()).tolist()
            # Use RapidFuzz to efficiently find the best match
            match = process.extractOne(
                toc_text, candidate_texts, scorer=fuzz.token_set_ratio
            )
            
            if match and match[1] >= min_score:
                matched_text = match[0]
                idx = candidates[candidates['text'].apply(
                    lambda txt: ''.join(txt.split('*')).upper()) == matched_text].index
                if not idx.empty:
                    matches.append({
                        'index': idx[0],
                        'level': level,
                        'text': f"\n{(level + 1) * '#'} {''.join(matched_text.split('*'))}\n",
                        'page': toc_page,
                        'similarity': match[1]/100
                    })

        return pd.DataFrame(matches)

    def _has_toc(self):
        return len(self.doc.get_toc(simple=False)) > 0

    def _xtract_toc(self, pretty=False) -> List[Tuple[int, str, int]] | str:
        toc_entries = self.doc.get_toc(simple=False)
        if pretty:
            lines = []
            for entry in toc_entries:
                level = entry[0]
                title = entry[1]
                page = entry[2]
                lines.append(
                    f"{(level + 1) * '#'} {' '.join(title.split('\n')).strip()} - (page {page})")
            return os.linesep.join(lines)
        return toc_entries

    def _apply_heading_heuristics(self, string: str) -> float:
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

        if not words or len(clean_str) < 3:
            return 0.0

        if word_count / token_count < 0.5:
            return 0.0

        # Length-based scoring
        if word_count <= 8:  # Ideal heading length
            score += 0.6
        elif word_count <= 15:
            score += 0.4
        else:  # Too long, likely paragraph
            for i in range(1, word_count - 15):
                score -= 0.2 * (i + 1)  # Increased penalty for very long text

        # Section number pattern (e.g., "1.", "1.2", "A.", "I.")
        if re.match(r'^(?:(?:\d+\.)+\d*\s|\d+\.\s|\d\s+[A-Z]|[A-Z]\.\s|[IVX]+\.\s)', clean_str):
            score += 0.9

        # Capitalization patterns
        score += self._caps_ratio(clean_str)
        if not words[0][0].isupper():  # First word *not* capitalized
            score -= 0.6

        # Penalize sentence-ending punctuation
        if self._has_end_punctuation(clean_str):
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

    def _compute_whitespace(self, lines:List[LineVector]):
        """Retrurns the whitespace below of a given line of text in the pdf"""
        def compute(prev, curr, nxt):
            upper_ws = abs(curr['bbox'][1] - prev['bbox'][-1])
            lower_ws = abs(curr['bbox'][-1] - nxt['bbox'][1])
            ret = upper_ws + lower_ws
            return upper_ws, lower_ws, ret
        
        # Clean lines
        clean_lines = []
        dirty_line_txt = {'<start_of_block>', '<end_of_block>', '</BR>'}
        for line in lines:
            if line['text'] in dirty_line_txt:
                continue
            clean_lines.append(line)

        for i in range(len(clean_lines)):
            prev:LineVector = clean_lines[i - 1] if i > 0 else None
            curr:LineVector = clean_lines[i]
            nxt:LineVector = clean_lines[i + 1] if i < len(clean_lines) - 1 else None

            if prev and nxt:
                _, lower_ws, _ = compute(prev, curr, nxt)
            else:
                _, lower_ws, _ = None, curr['bboxh'], curr['bboxh']

            curr['whitespace'] = lower_ws
        return clean_lines

    def _word_count(self, str: str):
        # Clean and prepare string
        clean_str = str.strip()
        words = re.findall(r'\b[a-zA-Z]+\b', clean_str)
        word_count = len(words)
        return word_count
    
    def _caps_ratio(self, str: str):
        ignore = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'in', 'nor', 'of', 'on', 'or', 'so', 'the', 'to', 'up', 'yet', 'over', 'with', 'without', 'into', 'onto', 'from', 'than', 'via'}
        clean_str = str.strip()
        words = re.findall(r'\b[a-zA-Z]+\b', clean_str)
        words = [word for word in words if word not in ignore]
        if not words:
            return 0.0
        caps_count = sum([1 for word in words if word[0].isupper()])
        caps_ratio = caps_count / len(words)
        if ' '.join(words).isupper():
            caps_ratio += 0.5
        return caps_ratio

    def _has_end_punctuation(self, str: str):
        pattern_match = re.search(r'[.,;:]$', str.strip())
        if pattern_match:
            return 1.0
        return 0.0

    def _compute_heading_entropy(self, block: List[Dict[str, Union[str, float, int]]]) -> List[Dict[str, Union[str, float, int]]]:
        def predict(title_score, weight, size, is_first):
            """Weighted sum of features."""
            if any([title_score == 0.0, is_first == 0.0]):
                return 0.0
            # return round(title_score + weight * 0.5 + size * 0.1, 2)
            return title_score + weight * 0.5

        # Identify potential heading lines (sliding window = 3)
        ignore = {'</BR>', '<start_of_block>'}
        for i in range(len(block) - 2):
            curr = block[i + 1]  # vector
            prefix = block[i]['text'].strip()  # string
            suffix = block[i + 2]  # vector

            # Ignore unimporant lines
            if curr['text'].strip() in ignore.union('<end_of_block>'):
                curr['entropy'] = 0.0
                continue
            
            # Note: Good candidate if preceded by </BR> or <start_of_block>
            if prefix in ignore:
                v0 = np.array([
                    curr.get('size', 0.0),
                    curr.get('weight', 0.0),
                ], dtype=np.float32)
                v1 = np.array([
                    suffix.get('size', 0.0),
                    suffix.get('weight', 0.0),
                ], dtype=np.float32)

                # Dot product similarity
                norm2 = round(np.linalg.norm(v0) ** 2, 2)
                dot = round(float(np.dot(v0, v1)), 2)
                sim = dot / norm2

                # If equal syntactically, treat current line as paragraph text
                if sim == 1.0:
                    # Note: We exclude fully consecutive weighted lines, since those are good headings candidates
                    if suffix.get('weight', 0.0):
                        curr['text'] = f"{curr['text']} {suffix['text']}"
                        curr['start'] = 1.0
                        suffix['text'] = ''
                    else:
                        curr['start'] = curr.get('start', 0.0)

                # If highly similar (i.e. not same) but weighted, treat both as a single heading candidate
                elif sim > 0.8 and suffix.get('weight', 0.0):
                    curr['start'] = 1.0
                    suffix['start'] = 1.0

                # Otherwise, mark it as a start (e.g., heading)
                else:
                    curr['start'] = 1.0
            else:
                # Not preceded by <start_of_block>
                curr['start'] = curr.get('start', 0.0)

            # Final heading entropy score
            curr['entropy'] = predict(
                curr.get('entropy', 0.0),
                curr.get('weight', 0.0),
                curr.get('size', 0.0),
                curr.get('start', 0.0)
            )

        return block

    def _handle_page(self, pno: int) -> List[LineVector]:
        '''
        wraps page with <start_of_page>...handle_blocks...</end_of_page> tags for llm understanding
        '''
        blocks = [self._merge_lines(block) for block in self.doc.load_page(pno).get_textpage().extractDICT().get('blocks', [])]
        mask_bboxes = [table.bbox for table in self.doc.load_page(pno).find_tables()]
        line_vectors = []
        for block in blocks:
            block_ = self._handle_block(block, pno, mask_bboxes)
            line_vectors.extend(block_)

        line_vectors = self._compute_whitespace(line_vectors)
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

    def _handle_line(self, line: List[Span], pno: int, mask_bboxes) -> LineVector:
        def process_txt(line):
            _bolded = False # represents the start of a bolded text **<text>
            bolded_ = False # represents the end of a bolded text <text>**
            txt = ''
            for span in line:
                match = is_bold(span=span)
                if match:
                    # Started run already
                    if _bolded:
                        _bolded = False
                        bolded_ = True
                    
                    # Need to start run
                    else:
                        _bolded = True
                        bolded_ = False
                
                if _bolded:
                    txt += '**' + span['text']
                elif bolded_:
                    txt += span['text'] + '**'
                else:
                    txt += span['text']
                
            # Return a single lined, md-styled text
            return txt.strip()
                

        def is_bold(span):
            # Don't consider weird unweighted items as a logic-breaking pivot
            # Note: '!' is a special character used by PyMuPDF for unknown ascii codes
            if span['text'].strip() in {'</BR>', '!', ''}:
                return True
            return any(bold_match in span['font'] for bold_match in ["Bold", "TB", "Medi", "CMB"])

        def is_italic(span):
            # Note '!' is a special character for unknown ascii codes
            if span['text'].strip() in {'</BR>', '!'}:
                return True
            return any(bold_match in span['font'] for bold_match in ["oblique", "CMTI", "CMMI", "Ital"])

        # Handle breaklines
        if line and not line[0]['text'].strip():
            line[0]['text'] = '</BR>'

        # Reduce...
        line_vector = LineVector()
        line_vector['text'] = process_txt(line=line)
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

    def _augment_line_vector(self, line_vector: LineVector, mask_bboxes: List[any]):
        line_vector['wc'] = self._word_count(''.join(line_vector['text'].split('*')).strip())
        line_vector['caps_ratio'] = self._caps_ratio(''.join(line_vector['text'].split('*')).strip())
        line_vector['weight'] = line_vector['bold'] + line_vector['italics']
        line_vector['ignore'] = self._ignore_bboxes(line_vector, mask_bboxes)
        line_vector['end_punctuation'] = self._has_end_punctuation(line_vector['text'])
        line_vector['bboxh'] = np.log(abs(line_vector['bbox'][1] - line_vector['bbox'][3]))
        line_vector['start'] = 0.0 # defines whether a line is the first in the block of text
        toi = ''.join(line_vector['text'].split('*')).strip()
        toi = ''.join(toi.split('</BR>')).strip()
        line_vector['entropy'] = self._apply_heading_heuristics(toi)
        if toi and toi[0] == '#':
            line_vector['text'] = f"`{toi}`"
        if line_vector['bold'] and line_vector['text'] != '</BR>':
            line_vector['text'] = f"**{toi}**"
        if line_vector['italics'] and line_vector['text'] != '</BR>':
            line_vector['text'] = f"*{line_vector['text']}*"
            line_vector['size'] -= 0.01
        return line_vector

    def _merge_lines(self, block: List) -> List:
        '''
        Merges lines by y0 (..if on the same column - in multi-column documents)
        '''
        # Only process text blocks
        if block['type'] != 0:
            return
        
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
                        true_lines[y0][-1]['bbox'][2] < doc_with /
                            2 and line['spans'][0]['bbox'][0] < doc_with / 2
                    ):
                        line['spans'][0]['text'] = f" {line['spans'][0]['text']}"
                        true_lines[y0].extend(line['spans'])

        return {'lines': list(true_lines.values())}

    def _ignore_bboxes(self, line_vector: LineVector, bboxes: List[any]) -> bool:
        x0, y0 = line_vector['origin']
        # Headers
        if y0 < 70:
            return True
        # Footers
        if y0 > self.doc[0].bound()[3] - 50:
            return True
        # Provided bboxes
        for bbox in bboxes:
            if x0 > bbox[0] and x0 < bbox[2] and y0 > bbox[1] and y0 < bbox[3]:
                return True
        return False

     ##################### Logistic Regression ############################

    def _augment_dfoc(self, df:pd.DataFrame):
        # Add bucket w.r.t. paragraph size (i.e. <, ==, >)
        def bucketize(d):
            p = df.loc[df['size'] != 0, 'size'].mode().iloc[0]
            if d < p:
                return 0.0
            if d == p:
                return 0.5
            if d > p:
                return 1.0
        
        df.loc[:, 'dist2par'] = df['size'].apply(bucketize)
        df.loc[df['entropy'] > 0, 'entropy'] = df['entropy'] + df['dist2par']
        return df

    ###############################################################################################################################

    def _predict(self):
        inference_data = self._get_inference_data(self.features)
        X = inference_data.drop(labels=['size', 'text', 'page'], axis=1)
        if not self.model:
            model_path = os.path.join(os.path.dirname(__file__), "model", "logreg.pkl")
            self.model = joblib.load(model_path)
        y_pred = self.model.predict(X)
        inference_data['label'] = y_pred
        ham = inference_data[inference_data['label'] == 1]
        return ham

    def _get_inference_data(self, features=[]):
        # Filter out zero entropy lines
        X_raw = self.dfoc[self.dfoc['ignore'] == False]
        X_raw = X_raw[X_raw['entropy'] > 0]
        # Select inference features
        X_inference = X_raw.loc[:, features or self.features]
        return X_inference
    
    def _get_training_data(self, features=[]):
        # Select training features
        X_train = self._get_inference_data(features=features + ['text'])
        # Get positive labels
        positive_idxs = self._predict().index
        # Assign labels
        X_train['label'] = 0
        X_train.loc[positive_idxs, 'label'] = 1
        return X_train