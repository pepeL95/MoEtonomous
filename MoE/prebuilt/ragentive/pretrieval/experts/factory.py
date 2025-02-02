from moe.prebuilt.ragentive.pretrieval.experts.repo import QueryAugmentationExpert, HydeExpert, Router

class PretrievalDirectory:
    Router:str = Router.__name__
    QueryAugmentationExpert:str = QueryAugmentationExpert.__name__
    HydeExpert:str = HydeExpert.__name__

class PretrievalFactory:
    @staticmethod
    def get(expert_name, agent=None):
        from moe.prebuilt.ragentive.pretrieval.strategies import QueryAugmentationStrategy, HydeStrategy, RouterStrategy
        
        if expert_name == PretrievalDirectory.Router:
            return Router(agent=agent, strategy=RouterStrategy())
        if expert_name == PretrievalDirectory.QueryAugmentationExpert:
            return QueryAugmentationExpert(agent=agent, strategy=QueryAugmentationStrategy())
        if expert_name == PretrievalDirectory.HydeExpert:
            return HydeExpert(agent=agent, strategy=HydeStrategy())
        else:
            raise ValueError(f'No expert by name of {expert_name} exists.')