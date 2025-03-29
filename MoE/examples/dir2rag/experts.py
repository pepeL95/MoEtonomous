import os
import time

from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
from dev_tools.enums.llms import LLMs
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
    

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