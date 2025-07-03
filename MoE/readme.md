# MoE (Mixture of Experts)

A powerful framework for building AI systems that dynamically route tasks to specialized experts using LangGraph and LangChain.

## What is MoE?

**MoE (Mixture of Experts)** is a sophisticated pattern where multiple AI experts collaborate to solve complex tasks. Instead of relying on a single AI model, MoE systems intelligently route different parts of a problem to specialized experts, then combine their outputs for optimal results.

### Key Benefits

- **🎯 Specialization**: Each expert focuses on specific domains or tasks
- **🔄 Dynamic Routing**: Intelligent decision-making about which expert to use
- **🧠 Context Awareness**: Maintains conversation history and state across experts
- **⚡ Scalability**: Easy to add new experts without changing existing ones
- **🔧 Flexibility**: Support for both autonomous (LLM-based) and deterministic routing

### When to Use MoE?

- **Complex Multi-Domain Tasks**: When a single AI can't handle all aspects of a problem
- **Specialized Workflows**: When different tasks require different expertise
- **Conversational AI**: Building chatbots that can handle diverse topics
- **RAG Systems**: Combining retrieval, reasoning, and generation experts
- **Research & Analysis**: Academic paper processing, data analysis, etc.

## Quick Start

```python
from moe.annotations.core import Expert, MoE, Autonomous
from dev_tools.enums.llms import LLMs
from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent

# Define your experts
@Expert()
class MathExpert:
    '''Expert in mathematical calculations and problem solving.'''
    agent = EphemeralNLPAgent(
        name='MathAgent',
        llm=LLMs.Gemini(),
        system_prompt='You are an expert in mathematics and problem solving.'
    )

@Expert()
class CodeExpert:
    '''Expert in programming and code generation.'''
    agent = EphemeralNLPAgent(
        name='CodeAgent',
        llm=LLMs.Gemini(),
        system_prompt='You are an expert in programming and code generation.'
    )

# Create your MoE
@MoE()
@Autonomous(LLMs.Gemini())
class SmartAssistant:
    '''AI assistant that can handle math and coding tasks.'''
    experts = [MathExpert(), CodeExpert()]

# Use it
assistant = SmartAssistant()
result = assistant.invoke({'input': 'Solve 2x + 5 = 15 and write Python code to verify it'})
```

## Installation

### Prerequisites

- **Python 3.8+** (recommended: Python 3.9+)
- **Git** for cloning the repository
- **API Keys** for your chosen LLM providers

> **💡 Note:** If you encounter metaclass conflicts with `langchain-google-genai`, ensure you're using a compatible Python environment. The framework has been tested with conda environments and works best with isolated Python environments.

### Core Dependencies

Install the required packages:

```bash
# Clone the repository
git clone <repository-url>
cd MoEtonomous

# Install core dependencies
pip install -r requirements.txt

# Install additional dependencies for advanced features
pip install langgraph langchain-google-genai langchain-huggingface sentence-transformers torch transformers
```

### Environment Configuration

Create a `.env` file in your project root with the following variables:

```bash
# Required: LLM API Keys
GOOGLE_API_KEY=your_google_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here  # Optional: for OpenAI models

# Required: LangChain
LANGCHAIN_API_KEY=your_langchain_api_key_here

# Optional: Local paths (adjust as needed)
SRC=/path/to/your/project/root
VDB_PATH=/path/to/your/vector/database
```

### Optional Dependencies

For advanced features, install additional packages:

```bash
# For RAG and vector operations
pip install chromadb umap-learn scikit-learn

# For PDF processing
pip install PyMuPDF rapidfuzz

# For JIRA integration
pip install jira

# For enhanced embeddings and cross-encoding
pip install sentence-transformers transformers torch
```

### Verification

Test your installation:

```python
# Quick verification script
from moe.annotations.core import Expert, MoE, Autonomous
from dev_tools.enums.llms import LLMs
from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent

# Test basic imports
print("✅ Core imports successful")

# Test LLM initialization
try:
    llm = LLMs.Gemini()
    print("✅ LLM initialization successful")
except Exception as e:
    print(f"❌ LLM initialization failed: {e}")

# Test agent creation
try:
    agent = EphemeralNLPAgent(
        name="TestAgent",
        llm=llm,
        system_prompt="You are a helpful assistant."
    )
    print("✅ Agent creation successful")
except Exception as e:
    print(f"❌ Agent creation failed: {e}")
```

### Troubleshooting

**Common Issues:**

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **API Key Issues**: Verify your `.env` file is in the project root
   ```bash
   ls -la .env  # Should show the file
   ```

3. **LangGraph Version**: Ensure compatibility
   ```bash
   pip install langgraph>=0.0.20
   ```

4. **CUDA/GPU Issues**: For GPU acceleration
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Metaclass Conflicts**: Use a compatible Python environment
   ```bash
   # Create a new conda environment
   conda create -n moe python=3.9
   conda activate moe
   pip install -r requirements.txt
   ```

## Table of Contents

- [MoE Folder Structure](#moe-folder-structure)
- [MoE State](#moe-state)
- [Building Experts](#expertspy)
- [Strategies](#strategypy)
- [Creating MoEs](#mainpy)
- [Examples](#examples)

---

## MoE Folder Structure

- When building your MoE, the following project structure is recommended:

``` Plaintext
custom_moe/
|
|____ experts.py
|
|____ strategies.py
|
|____ main.py
```

## MoE State

- The MoE manages a State that's defined as follows:

```python
class State(TypedDict):
    next: str # By default your router decides, but can enforce it in your strategy (i.e. state['next'] = '<Expert Name>')
    prev: str # the previous expert
    input: str # input to the MoE
    expert_input: str # internal expert input
    expert_output: str # internal expert output
    router_scratchpad: str
    ephemeral_mem: BaseMemory # memory for the run
    kwargs: dict # additional keys you may want to add in your MoE
```

## `experts.py`

- Should contain your expert classes
- Expert classes are classes annotated with the `@Expert` (or `@MoE` if your expert itself is an MoE)
- You **MUST** have a static `agent` under your class annotated with `@Expert`, which should extend `BaseAgent` or `Runnable` in general
- **Recommended:** *Use a factory pattern for ease of use when building your MoE*

```python
from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
from moe.annotations.core import Expert
from dev_tools.enums.llms import LLMs
from ...custom_moe.strategies import GenXpertStrategy

@Expert(GenXpertStrategy)
class GenXpert:
    '''Excellent expert on a wide range of topics. Default to this expert when not sure which expert to use.'''
    agent = EphemeralNLPAgent(
        name='GenAgent',
        llm=LLMs.Gemini(),
        system_prompt='You are an intelligent expert.',
    )

@Expert()
class WebSearchExpert:
    '''Expert in web search and information retrieval.'''
    agent = EphemeralNLPAgent(
        name='WebSearchAgent',
        llm=LLMs.Gemini(),
        system_prompt='You are an expert in web search and information retrieval.',
    )

################## RECOMMENDED ######################

class Factory:
    class Dir:
        GenXpert: str = 'GenXpert'
        WebSearchExpert: str = 'WebSearchExpert'
        # Add more experts as needed
    
    @staticmethod
    def get(expert_name: str):
        if expert_name == Factory.Dir.GenXpert:
            return GenXpert()
        if expert_name == Factory.Dir.WebSearchExpert:
            return WebSearchExpert()
        # Add more expert mappings as needed
        raise ValueError(f'No expert by name {expert_name} exists.')
```

## `strategy.py`

- Strategies define the behavior of the expert node in the mixture graph
- Strategies should inherit from `BaseExpertStrategy` or `BaseMoEStrategy`
- Strategies expose the `execute` method
- For example:

```python
from moe.base.strategies import BaseExpertStrategy

class GenXpertStrategy(BaseExpertStrategy):
    def execute(self, expert, state):
        output = expert.invoke({
            'input': state['expert_input'],
            # ... other template vars for the given expert
        })

        state['expert_output'] = output
        # Note: By default your MoE router chooses the next expert, but you can enforce it like state['next'] = '<expert_name>'
        return state

# ... other strategy classes ...

```

## `main.py`

- Regression test your MoE.
- Annotate your custom MoE class with `@MoE`
- Note: When using the `@MoE` annotation, you **MUST** have your `experts` array under your defined class.
- Note: When using the `@MoE` annotation, you **MUST** provide a router through:
- - 1 `@Autonomous(llm)` for an autonomous router.
- - 2 `@Deterministic(expert_name)` if your MoE is linear and you enforce next in all your strategies. Provides the starting point

```python
from dev_tools.enums.llms import LLMs
from moe.config.debug import Debug
from moe.annotations.core import MoE, Autonomous
from moe.default.strategies import DefaultMoEStrategy
from ...custom_moe.experts import Factory # Here is where the Factory comes in handy (e.g. imagine having many experts)


if __name__ == '__main__':
    
    @MoE(DefaultMoEStrategy)
    @Autonomous(LLMs.Gemini())
    class ChatMoE:
        '''MoE that provides Conversational and Websearch functionality'''
        experts = [
            Factory.get(expert_name=Factory.Dir.GenXpert),
            Factory.get(expert_name=Factory.Dir.WebSearchExpert),
        ]
    
    # Run
    try:
        chat = ChatMoE(verbose=Debug.Verbosity.low)
        user_input = input('user: ')
        state = chat.invoke({
                'input': user_input,
        })
        print(f"Response: {state.get('output', 'No output generated')}")
    except Exception as e:
        print(f"Error running MoE: {e}")
```
