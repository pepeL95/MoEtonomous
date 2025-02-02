from moe.prebuilt.ragentive.postrieval.experts.repo import RerankingExpert, ContextExpert, Router

class PostrievalDirectory:
    Router:str = Router.__name__
    RerankingExpert:str = RerankingExpert.__name__
    ContextExpert:str = ContextExpert.__name__

class PostrievalFactory:
    @staticmethod
    def get(expert_name, agent=None):
        from moe.prebuilt.ragentive.postrieval.strategies import RerankingStrategy, ContextStrategy, RouterStrategy

        if expert_name == PostrievalDirectory.Router:
            return Router(agent=agent, strategy=RouterStrategy())
        if expert_name == PostrievalDirectory.RerankingExpert:
            return RerankingExpert(agent=agent, strategy=RerankingStrategy())
        if expert_name == PostrievalDirectory.ContextExpert:
            return ContextExpert(agent=agent, strategy=ContextStrategy())
        else:
            raise ValueError(f'No expert by name of {expert_name} exists.')