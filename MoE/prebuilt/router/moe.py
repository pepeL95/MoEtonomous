from agents.config.debug import Debug
from moe.base.mixture import BaseMoE
from moe.prebuilt.router.strategies import RouterStrategy
from moe.prebuilt.router.experts.factory import ReActFactory, ReActDirectory


class AutonomousRouter(BaseMoE):
    def __init__(self, llm):
        super().__init__(
            name='Router',
            description='MoE that implements the ReAct framework for LLMs. It thinks, plans, and acts to non-naively fulfill a request.',
            router=ReActFactory.get(expert_name=ReActDirectory.Router, llm=llm),
            strategy=RouterStrategy(),
            verbose=Debug.Verbosity.quiet,
            experts=[
                ReActFactory.get(expert_name=ReActDirectory.IntentXtractor, llm=llm),
                ReActFactory.get(expert_name=ReActDirectory.PlanningXpert, llm=llm),
                ReActFactory.get(expert_name=ReActDirectory.SynthesisXpert, llm=llm),
            ],
        )
