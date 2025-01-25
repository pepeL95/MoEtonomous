from MoE.mixtures.ragentive.postrieval.experts.repo import RerankingExpert, ContextExpert, Router

class PostrievalDirectory:
    Router:str = Router.__name__
    RerankingExpert:str = RerankingExpert.__name__
    ContextExpert:str = ContextExpert.__name__

class PostrievalFactory:
    @staticmethod
    def get(expert_name, llm, prompt_parser=None):
        from MoE.mixtures.ragentive.postrieval.strategies import RerankingStrategy, ContextStrategy, RouterStrategy

        if expert_name == PostrievalDirectory.Router:
            return Router(llm=None, prompt_parser=None, strategy=RouterStrategy())
        if expert_name == PostrievalDirectory.RerankingExpert:
            return RerankingExpert(llm=None, prompt_parser=prompt_parser, strategy=RerankingStrategy())
        if expert_name == PostrievalDirectory.ContextExpert:
            return ContextExpert(llm=llm, prompt_parser=prompt_parser, strategy=ContextStrategy())
        else:
            raise ValueError(f'No expert by name of {expert_name} exists.')