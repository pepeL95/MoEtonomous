from MoE.mixtures.ragentive.pretrieval.experts.repo import QueryAugmentationExpert, HydeExpert, Router

class PretrievalDirectory:
    Router:str = Router.__name__
    QueryAugmentationExpert:str = QueryAugmentationExpert.__name__
    HydeExpert:str = HydeExpert.__name__

class PretrievalFactory:
    @staticmethod
    def get(expert_name, llm, prompt_parser=None):
        from MoE.mixtures.ragentive.pretrieval.strategies import QueryAugmentationStrategy, HydeStrategy, RouterStrategy
        
        if expert_name == PretrievalDirectory.Router:
            return Router(llm=llm, prompt_parser=prompt_parser, strategy=RouterStrategy())
        if expert_name == PretrievalDirectory.QueryAugmentationExpert:
            return QueryAugmentationExpert(llm=llm, prompt_parser=prompt_parser, strategy=QueryAugmentationStrategy())
        if expert_name == PretrievalDirectory.HydeExpert:
            return HydeExpert(llm=llm, prompt_parser=prompt_parser, strategy=HydeStrategy())
        else:
            raise ValueError(f'No expert by name of {expert_name} exists.')