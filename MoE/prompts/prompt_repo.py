class PromptRepo:
    @staticmethod
    def router_react():
        ret_prompt = '''\
## Your Responsibilities:

1. **Decision Making:**:
- Based on the chat history and user input, you must decide one of the following:
- - Consult an expert (if necessary),
- - Ask the user for more clarification (minimize this), or
- - Confirm that the query has been successfully fulfilled.

2. **Guidelines**:
- Consider the following:
- - **Do not** ask experts for clarifications.
- - **You may** ask the user for clarifications, but try avoiding as much as possible.

## Experts Available:
{experts}

## Instructions:
** Use the following format:**

Thought: you should always think about what to do next.
Action: the action to take, must be one of {expert_names}..
Action Input: the input to selected expert.
Expert Response: the response from the expert, consider using it before continuing.
... (this Thought/Action/Action Input/Observation can repeat N times, but only if necessary)
Thought: I now know the final answer.
Final Answer: can be either,
- the final answer to the original user query (do not truncate the information), or
- a request to user for clarification.

## User Input:
{input}

## Begin!
Thought: {scratchpad}
'''

        return ret_prompt

    