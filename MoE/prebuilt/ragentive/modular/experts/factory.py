from moe.prebuilt.ragentive.modular.experts.repo import Pretrieval, Retrieval, Postrieval, Router


class RagDirectory:
    Router: str = Router.__name__
    PretrievalMoE: str = Pretrieval.__name__
    Retrieval: str = Retrieval.__name__
    PostrievalMoE: str = Postrieval.__name__


class RagFactory:
    @staticmethod
    def get(expert_name: str, agent=None):
        from moe.prebuilt.ragentive.modular.strategies import PretrievalStrategy, PostrievalStrategy, RetrievalStrategy, RouterStrategy

        if expert_name == RagDirectory.Router:
            return Router(agent=agent, strategy=RouterStrategy())
        if expert_name == RagDirectory.PretrievalMoE:
            return Pretrieval(strategy=PretrievalStrategy()).build() # MoE
        if expert_name == RagDirectory.Retrieval:
            return Retrieval(agent=agent, strategy=RetrievalStrategy())
        if expert_name == RagDirectory.PostrievalMoE:
            return Postrieval(strategy=PostrievalStrategy()).build() # MoE

        raise ValueError(f'No expert by name {expert_name} exists.')
