class PromptRepo:
    @staticmethod
    def router_react():
        ret_prompt = '''\
## Your Responsibilities:

1. **Decision Making:**:
- Based on the context and user input, you must decide one of the following:
- - Consult an expert (if necessary),
- - Ask the user for more clarification (minimize this), or
- - Confirm that the query has been successfully fulfilled.

2. **Guidelines**:
- Consider the following:
- - **NEVER** ask experts for clarifications.
- - **YOU MAY** ask the user for clarifications, but avoid this as much as possible!!

## Experts Available:

{experts}

## Instructions:

** You MUST use the following format:**

Thought: generate an insightful thought about the current state.
Plan: break down thought into steps and determine what to do next in order to work towards achieving the goal.
Action: the action to take, must be exactly one of {expert_names}..
Action Input: the input to selected expert.
Expert Response: the response from the expert, consider using it before continuing.
... (this Thought/Plan/Action/Action Input/Expert Response may repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original user query.

## User Input:

{input}

## Begin!

Thought: {scratchpad}
'''

        return ret_prompt

    