from MoE.mixtures.arxiv.experts.repo import Router, QbuilderXpert, SearchXpert, SigmaXpert


class ArxivDirectory:
    Router: str = Router.__name__
    QbuilderXpert: str = QbuilderXpert.__name__
    SearchXpert: str = SearchXpert.__name__
    SigmaXpert: str = SigmaXpert.__name__


class ArxivFactory:
    @staticmethod
    def get(expert_name: str, agent=None):
        # Late import to avoid circular import issue
        from MoE.mixtures.arxiv.strategies import RouterStrategy, QueryStrategy, SearchStrategy, SigmaStrategy

        # Factory pattern
        if expert_name == ArxivDirectory.Router:
            return Router(agent=agent, strategy=RouterStrategy())
        if expert_name == ArxivDirectory.QbuilderXpert:
            return QbuilderXpert(agent=agent, strategy=QueryStrategy())
        if expert_name == ArxivDirectory.SearchXpert:
            return SearchXpert(agent=agent, strategy=SearchStrategy())
        if expert_name == ArxivDirectory.SigmaXpert:
            return SigmaXpert(agent=agent, strategy=SigmaStrategy())

        raise ValueError(f'No expert by name `{expert_name}` exists.')
