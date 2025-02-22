# coMoE uso la mierda esta?

- MoEs are an abstraction of a LangGraph spanning tree.
- Basically, we just:
- - Define our experts
- - Define their strategies
- - Add them to the MoE
- Experts can be instances of `BaseExperts` or even `BaseMoE`

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
- Expert classes are classes annotaded with the `@Expert` (or `@MoE` if your expert itself is an MoE)
- You **MUST** have a static `agent` under your class annotated with `@Expert`, which should extend `BaseAgent` or `Runnable` in general
- **Recommended:** *Use a factory pattern for ease of use when building your MoE*

```python
from agents.prebuilt.ephemeral_nlp_agent import EphemeralNLPAgent
from moe.annotations.core import Expert
from ...custom_moe.strategies import GenXpertStrategy

@Expert(GenXpertStrategy)
class GenXpert:
    '''Excellent expert on a wide range of topics. Default to this expert when not sure which expert to use.'''
    agent = EphemeralNLPAgent(
        name='GenAgent',
        llm=LLMs.Gemini(),
        system_prompt='You are an intelligent expert.',
    )

...

################## RECOMMENDED ######################

class Factory:
    class Dir:
        GenXpert: str = 'GenXpert'
        ...
    
    @staticmethod
    def get(expert_name: str):
        if expert_name == Factory.Dir.Router:
            return RouterMoE(llm=LLMs.Gemini()).build()
        if expert_name == Factory.Dir.GenXpert:
            return GenXpert()
        ...
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
- - 2 `@ForceFirst(expert_name)` if your MoE is linear, this provides the starting point.

```python
from dev_tools.enums.llms import LLMs
from moe.config.debug import Debug
from moe.base.mixture import MoEBuilder
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
            Factory.get(expert_name=Factory.Dir.WebSearchXpert),
        ]
    
    
    # Run
    chat = ChatMoE(verbose=Debug.Verbosity.low)
    user_input = input('user: ')
    state = chat.invoke({
            'input': user_input,
    })
```
