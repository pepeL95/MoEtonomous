from MoE.mixtures.arxiv.experts.repo import Router, QbuilderXpert, SearchXpert, SigmaXpert


class ArxivDirectory:
    Router: str = Router.__name__
    QbuilderXpert: str = QbuilderXpert.__name__
    SearchXpert: str = SearchXpert.__name__
    SigmaXpert: str = SigmaXpert.__name__


class ArxivFactory:
    @staticmethod
    def get(expert_name: str, llm=None, prompt_parser=None):
        # Late import to avoid circular import issue
        from MoE.mixtures.arxiv.strategies import RouterStrategy, QueryStrategy, SearchStrategy, SigmaStrategy

        # Factory pattern
        if expert_name == ArxivDirectory.Router:
            return Router(llm=llm, prompt_parser=prompt_parser, strategy=RouterStrategy())
        if expert_name == ArxivDirectory.QbuilderXpert:
            return QbuilderXpert(llm=llm, prompt_parser=prompt_parser, strategy=QueryStrategy())
        if expert_name == ArxivDirectory.SearchXpert:
            return SearchXpert(llm=llm, prompt_parser=prompt_parser, strategy=SearchStrategy())
        if expert_name == ArxivDirectory.SigmaXpert:
            return SigmaXpert(llm=llm, prompt_parser=prompt_parser, strategy=SigmaStrategy())

        raise ValueError(f'No expert by name `{expert_name}` exists.')
