import os
import time

from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
from dev_tools.enums.llms import LLMs
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

# Custom exception to signal rate limit issues.
class RateLimitError(Exception):
    pass

class APIKey:
    """
    A class representing an API key with rate-limiting counters and optional persistence of usage across sessions.
    """
    def __init__(self, name, key, rpm, tpm, rpd, persist_file=None):
        self.name = name
        self.key = key
        
        # If a quota is missing (None), treat it as unlimited.
        self.rpm_limit = rpm if rpm is not None else float('inf')
        self.tpm_limit = tpm if tpm is not None else float('inf')
        self.rpd_limit = rpd if rpd is not None else float('inf')
        
        # Usage counters
        self.rpm_used = 0
        self.tpm_used = 0
        self.rpd_used = 0
        
        # Reset times for each quota window
        self.rpm_reset_time = time.time() + 60      # 60-second window for RPM/TPM
        self.tpm_reset_time = time.time() + 60
        self.rpd_reset_time = time.time() + 86400     # 24-hour window for RPD

        # Optional file path where usage data is stored.
        self.persist_file = persist_file

        # If a persist file is provided, attempt to load usage from it.
        if self.persist_file:
            self._load_persistent_usage()

    def can_use(self, tokens=1):
        now = time.time()
        # Reset RPM counter if the window has expired.
        if now >= self.rpm_reset_time:
            self.rpm_used = 0
            self.rpm_reset_time = now + 60
        # Reset TPM counter if needed.
        if now >= self.tpm_reset_time:
            self.tpm_used = 0
            self.tpm_reset_time = now + 60
        # Reset RPD counter if needed.
        if now >= self.rpd_reset_time:
            self.rpd_used = 0
            self.rpd_reset_time = now + 86400

        # Check each quota. If any quota is exceeded, we cannot use this key.
        if self.rpm_used >= self.rpm_limit:
            return False
        if self.tpm_used + tokens > self.tpm_limit:
            return False
        if self.rpd_used >= self.rpd_limit:
            return False
        return True

    def record_call(self, tokens=1):
        self.rpm_used += 1
        self.tpm_used += tokens
        self.rpd_used += 1
        if self.persist_file:
            self._save_persistent_usage()

    def __repr__(self):
        return f"APIKey({self.name})"

    def _load_persistent_usage(self):
        """
        Loads usage counters and reset times from a JSON file.
        """
        import json, os
        if not os.path.exists(self.persist_file):
            return  # No saved usage yet
        try:
            with open(self.persist_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return  # If file is corrupted, ignore

        # The data should be a dict keyed by the API key string.
        # usage_info is a dictionary of counters.
        usage_info = data.get(self.key, None)
        if usage_info is not None:
            self.rpm_used = usage_info.get('rpm_used', 0)
            self.tpm_used = usage_info.get('tpm_used', 0)
            self.rpd_used = usage_info.get('rpd_used', 0)
            self.rpm_reset_time = usage_info.get('rpm_reset_time', time.time() + 60)
            self.tpm_reset_time = usage_info.get('tpm_reset_time', time.time() + 60)
            self.rpd_reset_time = usage_info.get('rpd_reset_time', time.time() + 86400)

    def _save_persistent_usage(self):
        """
        Saves usage counters and reset times to a JSON file.
        """
        if not self.persist_file:
            return  # No persistence enabled

        import json, os
        # Load existing data if any
        if os.path.exists(self.persist_file):
            try:
                with open(self.persist_file, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            data = {}

        data[self.key] = {
            'rpm_used': self.rpm_used,
            'tpm_used': self.tpm_used,
            'rpd_used': self.rpd_used,
            'rpm_reset_time': self.rpm_reset_time,
            'tpm_reset_time': self.tpm_reset_time,
            'rpd_reset_time': self.rpd_reset_time,
        }

        try:
            with open(self.persist_file, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # If we fail to save, ignore


class KeyRotator:
    def __init__(self, api_keys):
        self.keys = api_keys  # List of APIKey instances.
        self.index = 0

    def get_key(self, tokens=1):
        # Try each key once; if none are available, raise an error.
        for _ in range(len(self.keys)):
            key_obj = self.keys[self.index]
            # Advance the index (round-robin).
            self.index = (self.index + 1) % len(self.keys)
            # Check if the key can be used with the given token cost.
            if key_obj.can_use(tokens=tokens):
                return key_obj
        # All keys are exhausted.
        raise RateLimitError("No available API key at the moment.")


class KeyRotatorAgent:
    def __init__(self, key_rotator, agent):
        self.key_rotator = key_rotator
        self.agent = agent

    def invoke(self, agen_input, tokens=1, max_retries=3):
        attempts = 0
        while attempts < max_retries:
            try:
                key_obj = self.key_rotator.get_key(tokens=tokens)
                key_obj.record_call(tokens=tokens)
                os.environ['GOOGLE_API_KEY'] = key_obj.key  # If relevant to your environment
                response = self.agent.invoke(agen_input)
                return response

            except RateLimitError:
                # All keys are exhausted?
                raise Exception("All API keys are currently exhausted. Please wait until a quota resets.")
            except Exception as e:
                error_str = str(e)
                if "429 You exceeded your current quota" in error_str:
                    print(
                        "Key is out of sync with the actual API usage. Checking whether daily usage is maxed "
                        "or if we should wait for a minute reset."
                    )
                    # If the daily usage is exceeded, rotate keys
                    if key_obj.rpd_used >= key_obj.rpd_limit:
                        print("Daily limit reached. Rotating to a different key...")
                        self.key_rotator.invalidate_key(key_obj)
                        attempts += 1  # try again with another key
                    else:
                        # Just a per-minute or token-per-minute limit. Wait for the next reset.
                        # We'll wait for whichever reset time is sooner between rpm and tpm.
                        from time import sleep, time
                        now = time()
                        # How many seconds until the next reset?
                        wait_for_rpm = key_obj.rpm_reset_time - now
                        wait_for_tpm = key_obj.tpm_reset_time - now
                        time_until_reset = max(0, min(wait_for_rpm, wait_for_tpm))

                        print(f"Waiting {time_until_reset:.1f} seconds for the minute window to reset...")
                        sleep(time_until_reset)
                        attempts += 1  # try again with the same key
                else:
                    # Handle other exceptions as before
                    print(f"Error invoking agent (attempt {attempts + 1}/{max_retries}): {e}")
                    attempts += 1

        # If we've exhausted all attempts, raise an exception.
        raise Exception(f"Invocation failed after {max_retries} retries.")
    

semex = EphemeralNLPAgent(
    name='ThoughtXpert',
    llm=LLMs.Gemini2FlashLite(),
    prompt_template=(
        '## Instructions'
        'You are a highly analytical and methodical reasoning agent. Your task is to carefully examine every user query and produce '
        'a comprehensive, step-by-step explanation of your reasoning process. Instead of providing a final answer to the query, your '
        'output should consist solely of your detailed chain-of-thought. This chain-of-thought should include:\n'
        '1.	Query Understanding: A clear breakdown of what the query is asking.\n'
        '2.	Decomposition: Identification of key components, underlying assumptions, and any ambiguities.\n'
        '3.	Step-by-Step Reasoning: A thorough, sequential explanation of how you analyze each part of the query, including any considerations or alternative interpretations.\n'
        '4.	Verification: An overview of how you verify and validate your reasoning steps.\n\n'
        'Your entire response should be just this detailed reasoning process, without any concluding final answer or summary.'
        '## User Query\n'
        '{input}'
    ),
)

jobx = EphemeralNLPAgent(
    name='JobPostXpert',
    llm=LLMs.Gemini2FlashLite(),
    prompt_template=(
        '## Instructions\n'
        "You are an analysis expert. Given a noisy job description, as input, the objective is to extract the following:\n"
        "- Job Title\n"
        "- Job Description\n"
        "- Company info\n"
        "- What product is being developed?\n\n"
        "**Output Format**:\n"
        """
        {{
            "title": <"Job Title">,
            "description": <"Job description">,
            "company": <"Company information">,
            "product": <"Discussion of the product begin developed">
        }}
        """
        "**Note**:\n"
        "Respond only with the suggested output format, as JSON. Ommit any preamble, or prologue.\n\n"
        '## User Query\n'
        '{input}'
    ),
    output_parser=JsonOutputParser(),
)

abstract_highlighter = EphemeralNLPAgent(
    name='AbstractHighlightsXpert',
    llm=LLMs.Gemini2FlashLite(),
    prompt_template=(
        'You are an expert at analyzing research paper abstracts and extracting the most important highlights.\n'
        'For each abstract, identify and extract:\n'
        '1. The main research problem or objective\n'
        '2. Key methodology or approach\n'
        '3. Most significant findings/results\n'
        '4. Important conclusions or implications\n\n'
        'Format your response as a bulleted list with these categories.\n'
        'Be concise but preserve important technical details and numerical results.\n'
        'Focus on what makes this research novel or significant.\n\n'
        '**Input**\n'
        '{input}'
    )
)

abstract_highlite = EphemeralNLPAgent(
    name='AbstractHighlightsXpert',
    llm=LLMs.Gemini2FlashLite(),
    system_prompt='You are an expert at analyzing research paper abstracts and extracting the most important highlights.\n',
    prompt_template=(
        '**Instructions**\n'
        'Provide the highlights of the following paper extracted from the Arxiv repository.\n'
        'Return no more than 5 very concise and direct bullet points.\n'
        '**Input**\n'
        '{input}'
    )
)


section_summarizer = EphemeralNLPAgent(
    name='SectionSummarizer',
    llm=LLMs.Gemini2FlashLite(),
    prompt_template=(
        'Read the following section of text carefully. The text is a section of a larger document.\n'
        'Your task is to first extract the key highlights—these are the most significant points, findings, or insights presented in the text.\n'
        "You will be provided with the section's title, a table of contents, and text content.\n"
        '**Note**: The text context provided may contain more than the target section.\n'
        'Extract highlights, only from the target section (given by the section title).\n'
        'Include relevant context, as well as any broader consequences or avenues for further inquiry that arise from the highlight.\n\n'
        'The response should be comprehensive and analytical, ensuring that the extracted highlights are supported by a deep, thoughtful discussion.\n'
        'Aim for clarity and precision, and use logical arguments and evidence where possible to back your analysis.\n'
        'Ommit any preamble, or prologue.\n\n'
        '**Input**\n'
        '{input}'
    )
)

section_synth = EphemeralNLPAgent(
    name='SectionSynthesizer',
    llm=LLMs.Gemini2FlashLite(),
    prompt_template=(
        'Read the following section of text carefully. The text is a collection of highlights from a section of a larger document.\n'
        'Your task is to condense the highlights into a concise and accurate synthesis.\n'
        'Aim for clarity, precision, and efficiency.\n'
        'Ommit any preamble, or prologue.\n\n'
        '**Input**\n'
        '{input}'
    )
)

metadata_xtractor = EphemeralNLPAgent(
    name='MetadataExtractor',
    llm=LLMs.Gemini2FlashLite(),
    output_parser=JsonOutputParser(),
    prompt_template=(
        'Extract the following metadata from the provided text if it exists.\n\n'
        '**Metadata**\n'
        'title: Title of the document\n'
        'authors: Authors of the document\n'
        'date: Date of the document\n\n'
        '**Input**\n'
        '{input}\n\n'
        '**Output**\n'
        'Output only the metadata in the following format (if some metadata is not present, '
        'return null for the corresponding value).\n'
        '{{'
            '"title": <"Title of the document" or null>, '
            '"authors": <"Authors of the document" or null>, '
            '"date": <"Date of the document" or null>'
        '}}'
    )
)

references_extractor = EphemeralNLPAgent(
    name='ReferencesExtractor',
    llm=LLMs.Gemini2FlashLite(),
    prompt_template=(
        'Extract the references from the provided text.\n\n'
        'Your task is to identify and return the **text boundaries** of the References section within a provided text excerpt. \n'
        'The excerpt may include multiple sections from a larger document.\n'
        'Instructions:\n'
        '1. Locate the **start** of the References section, typically marked by a header such as "References", "Bibliography", or a similar heading.\n'
        '2. Determine the **end boundary** of the References section, which is typically:\n'
        '- The end of the document, or\n'
        '- The start of another section (e.g., "Appendix", "Acknowledgments") if present.\n'
        'Output Format:\n'
        '- Return the exact start and end markers, formatted as JSON, with the following schema:\n'
        '{{'
            '"start": <"The text of the line that marks the beginning (e.g., "References")">, '
            '"end": <"The text of the final line within the References section (e.g., the last bibliographic entry)">'
        '}}\n'
        'Note: If no References section is present, respond with "{{"start": None, "end": None}}".\n'
        'This information will be used to programmatically extract the References section in follow-up tasks.\n\n'
        '**Input**\n'
        '{input}'
    ),
    output_parser=JsonOutputParser(),
)