from moe.prebuilt.router.experts.repo import Router, IntentXtractor, PlanningXpert, SynthesisXpert


class ReActDirectory:
    Router: str = Router.__name__
    IntentXtractor: str = IntentXtractor.__name__
    PlanningXpert: str = PlanningXpert.__name__
    SynthesisXpert: str = SynthesisXpert.__name__
    ActionExecXpert: str = 'ActionExecXpert'


class ReActFactory:
    @staticmethod
    def get(expert_name: str, llm):
        from moe.prebuilt.router.strategies import PlanningStrategy, SynthesisStrategy, IntentXtractStrategy, InnerStrategy
        
        if expert_name == ReActDirectory.Router:
            return Router(strategy=InnerStrategy())
        if expert_name == ReActDirectory.IntentXtractor:
            return IntentXtractor(llm=llm, strategy=IntentXtractStrategy())
        if expert_name == ReActDirectory.PlanningXpert:
            return PlanningXpert(llm=llm, strategy=PlanningStrategy())
        if expert_name == ReActDirectory.SynthesisXpert:
            return SynthesisXpert(llm=llm, strategy=SynthesisStrategy())
        if expert_name == ReActDirectory.ActionExecXpert:
            return None
        raise ValueError(f'No expert by name {expert_name} exists.')
