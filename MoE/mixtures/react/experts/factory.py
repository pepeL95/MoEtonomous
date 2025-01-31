from MoE.mixtures.react.experts.repo import Router, IntentXtractor, PlanningXpert, SynthesisXpert


class ReActDirectory:
    Router: str = Router.__name__
    IntentXtractor: str = IntentXtractor.__name__
    PlanningXpert: str = PlanningXpert.__name__
    SynthesisXpert: str = SynthesisXpert.__name__
    ActionExecXpert: str = 'ActionExecXpert'


class ReActFactory:
    @staticmethod
    def get(expert_name: str, agent=None):
        from MoE.mixtures.react.strategies import PlanningStrategy, SynthesisStrategy, IntentXtractStrategy, RouterStrategy
        
        if expert_name == ReActDirectory.Router:
            return Router(strategy=RouterStrategy())
        if expert_name == ReActDirectory.IntentXtractor:
            return IntentXtractor(agent=agent, strategy=IntentXtractStrategy())
        if expert_name == ReActDirectory.PlanningXpert:
            return PlanningXpert(agent=agent, strategy=PlanningStrategy())
        if expert_name == ReActDirectory.SynthesisXpert:
            return SynthesisXpert(agent=agent, strategy=SynthesisStrategy())
        if expert_name == ReActDirectory.ActionExecXpert:
            return None
        raise ValueError(f'No expert by name {expert_name} exists.')
