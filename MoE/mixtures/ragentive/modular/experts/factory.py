from MoE.mixtures.ragentive.modular.experts.repo import Pretrieval, Retrieval, Postrieval, Router


class RagDirectory:
    Router: str = Router.__name__
    PretrievalMoE: str = Pretrieval.__name__
    Retriever: str = Retrieval.__name__
    PostrievalMoE: str = Postrieval.__name__


class RagFactory:
    @staticmethod
    def get(expert_name: str, retriever=None, llm=None, prompt_parser=None):
        from MoE.mixtures.ragentive.modular.strategies import PretrievalStrategy, PostrievalStrategy, RetrievalStrategy, RouterStrategy

        if expert_name == RagDirectory.Router:
            return Router(llm=llm, prompt_parser=prompt_parser, strategy=RouterStrategy())
        if expert_name == RagDirectory.PretrievalMoE:
            return Pretrieval(llm=llm, prompt_parser=prompt_parser, strategy=PretrievalStrategy())
        if expert_name == RagDirectory.Retriever:
            return Retrieval(llm=llm, prompt_parser=prompt_parser, strategy=RetrievalStrategy())
        if expert_name == RagDirectory.PostrievalMoE:
            return Postrieval(llm=llm, prompt_parser=prompt_parser, strategy=PostrievalStrategy())

        raise ValueError(f'No expert by name {expert_name} exists.')
