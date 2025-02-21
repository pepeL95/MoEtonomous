import functools
from collections import OrderedDict
from typing import Any, TypedDict, List

from langgraph.graph import END, StateGraph

from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from moe.config.debug import Debug
from moe.base.expert import BaseExpert
from moe.base.strategies import BaseMoEStrategy
from moe.default.strategies import DefaultMoEStrategy

from dev_tools.utils.clifont import CLIFont, print_bold


class MoEBuilder:
    def __init__(self):
        self.strategy = DefaultMoEStrategy()

    def set_name(self, name):
        self.name = name
        return self

    def set_description(self, description):
        self.description = description
        return self

    def set_router(self, router):
        self.router = router
        return self

    def set_experts(self, experts):
        self.experts = experts
        return self

    def set_strategy(self, strategy):
        self.strategy = strategy
        return self

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity
        return self

    def build(self):
        return BaseMoE(
            name=self.name,
            description=self.description,
            router=self.router,
            experts=self.experts,
            strategy=self.strategy,
            verbose=self.verbosity,
        ).build()


class BaseMoE:
    FINISH = '__end__'

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
            name: str,
            router: BaseExpert | Any,  # : BaseExpert | MoE
            experts: List[BaseExpert],
            description: str,
            strategy: BaseMoEStrategy,
            verbose: Debug.Verbosity = Debug.Verbosity.quiet) -> None:
        self.experts = OrderedDict()  # will be {expert_name: expert}
        for expert in experts:
            self.experts[expert.name] = expert
        self.name = name
        self.description = description
        self._graph = None
        self.verbose = verbose
        self.router = router
        self.strategy = strategy
        self._FINISH = '__end__'

        self.build()


####################################################### Class Methods #############################################################


    @classmethod
    def _extract_router_user_messages(cls, chat_history: ChatMessageHistory):
        '''Sliding window chat history'''
        history = []
        for mssg in chat_history.messages:
            if mssg.role == 'user' or mssg.role == 'assistant':
                history.append(mssg)
        return history[-5:]

    @classmethod
    def _parse_action_and_input(cls, text):
        first_split = text.split('\nAction:')[-1]
        second_split = first_split.split('\nAction Input:')
        if len(second_split) == 2:
            return second_split[0].strip(), second_split[1].strip()
        else:
            return None, None

    @classmethod
    def _parse_final_answer(cls, text):
        first_split = text.split('\nFinal Answer:')
        if first_split:
            return '__end__', first_split[-1].strip()

    @classmethod
    def _parse_router_output(cls, text):
        xpert, xpert_input = cls._parse_action_and_input(text)
        if all([xpert, xpert_input]):
            return xpert, xpert_input

        xpert, xpert_input = cls._parse_final_answer(text)
        if all([xpert, xpert_input]):
            return xpert, xpert_input

        raise ValueError(
            'Neither \'Expert Response\' or \'Final Answer\' was found')

##################################################################################################################################

    def _create_router_node(self, state: State):
        # All keys in State are required*
        required_keys = {'input', 'prev', 'next', 'router_scratchpad', 'expert_output', 'ephemeral_mem'}
        assert required_keys.issubset(state.keys(
        )), f'You are missing at least one of the following required keys {required_keys}'

        # Router is required
        if not self.router:
            raise ValueError(
                '`self.router` cannot be None. Make sure you initialize it in your specific mixture __init__() method.')

        # Router is of valid type
        if not isinstance(self.router, (BaseExpert, BaseMoE)):
            raise TypeError(f"Router must be an instance of Union[MoE, Expert]. Got {type(self.router)}")

        # Next state must be valid
        if state['next'] not in self.experts.keys() | {self.router.name, '__end__'}:
            raise ValueError(f'Next state is not defined in the mixture. Must be one of {self.experts.keys()}, but got {state['next']}')

        # Avoid infinite loops
        if state['next'] == state['prev'] or state['next'] is None:
            state['next'] = self.router.name

        # If xpert decided where to go next, go there
        if state['next'] != self.router.name:
            return state

        # Build scratchpad
        if state["expert_output"]:
            state["router_scratchpad"] += f"\n{state['prev']} Response: {state["expert_output"]}"

        # Call router
        output_map = self.router.execute_strategy({
            # 'chat_history': self._extract_router_user_messages(state['ephemeral_mem']),
            'experts': [f'{expert.name}: {expert.description}' for expert in self.experts.values()],
            'expert_names': [f'{expert}' for expert in self.experts.keys()],
            'input': state['input'],
            # Prev scratchpad + expert response (if any)
            'scratchpad': state["router_scratchpad"],
            'previous_expert': state['prev']
        })

        # Update scratchpad if needed
        # Prev scratchpad + expert response (compressed)
        state['router_scratchpad'] = output_map.get('router_scratchpad', state['router_scratchpad'])

        # Extract expert and expert input
        output = output_map['expert_output']
        xpert_name, xpert_input = self._parse_router_output(output)  # New plan + action + action input

        # Debug verbosity
        if self.verbose is Debug.Verbosity.high:
            print_bold(f'Scratchpad: {CLIFont.blue}{state['router_scratchpad']}')
            print_bold(f'Output: {CLIFont.blue}{output_map}\n')

        # Final answer?
        if xpert_name == '__end__' or xpert_name == 'USER':
            return {"next": '__end__', "expert_output": xpert_input, "router_scratchpad": output_map, "prev": self.router.name}

        # Sanity check
        if xpert_name not in self.experts.keys():
            raise ValueError(f"""`self.router` must return one of {
                             self.experts.keys()}. But got `{xpert_name}` instead.""")

        # Update state
        state["next"] = xpert_name
        state["expert_input"] = xpert_input
        state["prev"] = self.router.name
        state["router_scratchpad"] += f'\n{output}'

        return state

    def _create_expert_node(self, state: State, xpert: BaseExpert) -> dict:
        # The following keys in State are required*
        required_keys = {'input', 'prev', 'next', 'expert_input', 'expert_output', 'ephemeral_mem', 'kwargs'}
        assert required_keys.issubset(state.keys()), f'You are missing at least one of the following required keys {required_keys}'

        if self.verbose > Debug.Verbosity.quiet:
            print_bold(f'Result: {CLIFont.purple}Calling `{xpert.name}` with input `{state['expert_input']}`\n')

        update: dict = xpert.execute_strategy(state)

        xpert_in = HumanMessage(content=state['expert_input'], role=state['prev'])
        xpert_out = AIMessage(content=update['expert_output'], role=xpert.name)

        update['prev'] = xpert.name
        update['ephemeral_mem'].add_messages([xpert_in, xpert_out])
        # Input to expert_(t+1) = expert_t
        update['expert_input'] = update['expert_output']

        # Debug verbosity
        if self.verbose > Debug.Verbosity.quiet:
            print_bold(f'{xpert.name} Responded: {CLIFont.light_green}`{state['expert_output']}`{CLIFont.reset}`\n')
            print_bold(f'Next: {CLIFont.purple}Calling `{state['next']}`\n')

        return update

    def execute_strategy(self, input):
        return self.strategy.execute(self, input)

    def build(self):
        if not self.router:
            raise ValueError(
                'gate_keeper cannot be None. Make sure you initialize it in your specific mixture __init__() method.')

        # Define mixture state
        convo_team = StateGraph(BaseMoE.State)

        # Add experts
        convo_team.add_node(self.router.name, self._create_router_node)
        for name, xpert in self.experts.items():
            convo_team.add_node(name, functools.partial(
                self._create_expert_node, xpert=xpert))

        # Add simple edges
        for expert_name in self.experts:
            convo_team.add_edge(expert_name, self.router.name)

        # Add conditional edges
        conditional_edges = {'__end__': END}
        conditional_edges.update(
            {expert_name: expert_name for expert_name in self.experts.keys()})
        convo_team.add_conditional_edges(
            self.router.name,
            lambda s: s['next'],
            conditional_edges
        )

        # Build the mixture of experts
        convo_team.set_entry_point(self.router.name)
        self._graph = convo_team.compile()
        return self

    def invoke(self, input: dict):
        assert 'input' in input, f"Missing required `input` key when invoking {
            self.name}."

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
