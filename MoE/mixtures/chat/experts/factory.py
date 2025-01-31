from MoE.mixtures.chat.experts.repo import Router, GenXpert, WebSearchXpert


class ChatDirectory:
    Router: str = Router.__name__
    GenXpert: str = GenXpert.__name__
    WebSearchXpert: str = WebSearchXpert.__name__


class ChatFactory:
    @staticmethod
    def get(expert_name: str, agent=None):
        from MoE.mixtures.chat.strategies import GenXpertStategy, WebSearchStrategy, RouterStrategy

        if expert_name == ChatDirectory.Router:
            return Router(strategy=RouterStrategy()).build()  # MoE
        if expert_name == ChatDirectory.GenXpert:
            return GenXpert(agent=agent, strategy=GenXpertStategy())
        if expert_name == ChatDirectory.WebSearchXpert:
            return WebSearchXpert(agent=agent, strategy=WebSearchStrategy())

        raise ValueError(f'No expert by name {expert_name} exists.')
