import functools
from typing import TypedDict, List
from collections import OrderedDict

from langgraph.graph import END, StateGraph

from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from MoE.base.router import Router
from MoE.base.expert import Expert
from MoE.config.debug import Debug

from dev_tools.utils.clifont import CLIFont, print_bold


class MoE:
    class State(TypedDict):
        next: str
        prev: str
        input: str
        expert_input: str
        expert_output: str
        router_scratchpad: str
        ephemeral_mem: BaseMemory
        kwargs: dict

    def __init__(
        self,
    name:str,
    router:Router,
    experts:List[Expert],
    description:str=None,
    verbose:Debug.Verbosity=Debug.Verbosity.quiet) -> None:
        self.experts = OrderedDict() # {expert_name: expert}
        for expert in experts:
            self.experts[expert.name] = expert
        self.name = name
        self.description = description
        self._graph = None
        self.verbose = verbose
        self.router = router

####################################################### Class Methods #############################################################
    
    @classmethod
    def _extract_router_user_messages(cls, chat_history:ChatMessageHistory):
        '''Sliding window chat history'''
        history = []
        for mssg in chat_history.messages:
            if mssg.role == 'user' or mssg.role == 'assistant':
                history.append(mssg)
        return history[-5:]

    @classmethod
    def _parse_action_and_input(cls, text):
        first_split = text.split('\nAction: ')[-1]
        second_split = first_split.split('\nAction Input: ')
        if len(second_split) == 2:
            return second_split[0].strip(), second_split[1].strip()
        else:
            return None, None
        
    @classmethod
    def _parse_final_answer(cls, text):
        first_split = text.split('\nFinal Answer:')
        if first_split:
            return 'END', first_split[-1].strip()
    
    @classmethod
    def _parse_router_output(cls, text):
        xpert, xpert_input = cls._parse_action_and_input(text)
        if all([xpert, xpert_input]):
            return xpert, xpert_input

        xpert, xpert_input = cls._parse_final_answer(text)
        if all([xpert, xpert_input]):
            return xpert, xpert_input
        
        raise ValueError('Neither \'Expert Response\' or \'Final Answer\' was found')

##################################################################################################################################

    def _create_router_node(self, state: State):
        # All keys in State are required*
        required_keys = {'input', 'prev', 'next', 'router_scratchpad', 'expert_output', 'ephemeral_mem'}
        assert required_keys.issubset(state.keys()), f'You are missing at least one of the following required keys {required_keys}'

        # Router is required
        if not self.router:
            raise ValueError('`self.router` cannot be None. Make sure you initialize it in your specific mixture __init__() method.')

        # Next state must be valid
        if state['next'] not in self.experts:
            raise ValueError(f'Next state is not defined in the mixture. Must be one of {self.experts.keys()}, but got {state['next']}')

        # Avoid infinite loops
        if state['next'] == state['prev']:
            state['next'] = self.router.name
        
        # If xpert decided where to go next, go there
        if state['next'] != self.router.name:
            return state

        # Build scratchpad
        scratchpad = state["router_scratchpad"]
        if state["expert_output"]:
            scratchpad = scratchpad + f"\nExpert Response: {state["expert_output"]}"
        
        # Call router
        output = self.router.invoke({
            'chat_history': self._extract_router_user_messages(state['ephemeral_mem']),
            'experts': [f'{expert.name}: {expert.description}' for expert in self.experts.values()],
            'expert_names': self.experts.keys(),
            'input': state['input'],
            'scratchpad': scratchpad,
        })

        # Extract expert and expert input
        xpert_name, xpert_input = self._parse_router_output(output)
        
        # Debug verbosity
        if self.verbose is Debug.Verbosity.low:
            print_bold(f'Result: {CLIFont.purple}Calling `{xpert_name}` with input `{xpert_input}`\n')
            
        if self.verbose is Debug.Verbosity.high:
            print_bold(f'Scratchpad: {CLIFont.blue}{scratchpad}')
            print_bold(f'Output: {CLIFont.blue}{output}\n')
            print_bold(f'Result: {CLIFont.purple}Calling `{xpert_name}` with input `{xpert_input}`\n')
            
        # Final answer?
        if xpert_name == 'END':
            return {"next": 'END', "expert_output": xpert_input, "router_scratchpad": output, "prev": self.router.name}

        # Sanity check
        if xpert_name not in self.experts:
            raise ValueError(f"""`self.router` must return one of {self.experts.keys()}. But got `{xpert_name}` instead.""")

        # Update state
        state["next"] = xpert_name
        state["expert_input"] = xpert_input
        state["prev"] = self.router.name
        state["router_scratchpad"] = output

        return state
    
    def _create_expert_node(self, state:State, xpert:Expert) -> dict:
        # All keys in State are required*
        required_keys = {'input', 'prev', 'next', 'expert_input', 'expert_output', 'ephemeral_mem', 'kwargs'}
        assert required_keys.issubset(state.keys()), f'You are missing at least one of the following required keys {required_keys}'
        
        update:dict = self.define_xpert_impl(state=state, xpert=xpert)
        
        xpert_in = HumanMessage(content=state['expert_input'], role=state['prev'])
        xpert_out = AIMessage(content=update['expert_output'], role=xpert.name)
        
        update['prev'] = xpert.name
        update['ephemeral_mem'].add_messages([xpert_in, xpert_out])
        
        # Debug verbosity
        if self.verbose > Debug.Verbosity.quiet:
            print_bold(f'Expert Responded: {CLIFont.light_green}`{state['expert_output']}`{CLIFont.reset}`\n')
            print_bold(f'Next: {CLIFont.purple}Calling `{state['next']}`\n')

        return update

    def build_MoE(self):
        if not self.router:
            raise ValueError('gate_keeper cannot be None. Make sure you initialize it in your specific mixture __init__() method.')
        
        # Define mixture state
        convo_team = StateGraph(MoE.State)

        # Add experts
        convo_team.add_node(self.router.name, self._create_router_node)
        for name, xpert in self.experts.items():
            convo_team.add_node(name, functools.partial(self._create_expert_node, xpert=xpert))

        # Add simple edges
        for expert_name in self.experts:
            convo_team.add_edge(expert_name, self.router.name)

        # Add conditional edges
        conditional_edges = {'END': END}
        conditional_edges.update({expert_name: expert_name for expert_name in self.experts.keys()})
        convo_team.add_conditional_edges(
            self.router.name,
            lambda s : s['next'],
            conditional_edges
        )
        
        # Build the mixture of experts
        convo_team.set_entry_point(self.router.name)
        self._graph = convo_team.compile()
        return self

    def define_xpert_impl(self, state:State, xpert:Expert) -> dict:
        '''Override this method in your custom MoE to include the logic for each expert'''
        raise NotImplementedError('Gotta implement \'define_xpert_impl\' in your custom MoE')

    def invoke(self, input:dict):
        assert 'input' in input, f"Missing required `input` key when invoking {self.name}."

        # Set up defaults
        input.setdefault("next", self.router.name)
        input.setdefault("prev", None)
        input.setdefault("expert_input", None)
        input.setdefault("expert_output", None)
        input.setdefault("kwargs", {})
        input.setdefault("router_scratchpad", "")
        input.setdefault("ephemeral_mem", ChatMessageHistory())
        
        # Invoke graph
        return self._graph.invoke(input)