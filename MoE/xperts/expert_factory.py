from MoE.xperts.expert_repo import ExpertRepo
from dev_tools.enums.llms import LLMs

class ExpertFactory:
    class Directory:
        Router:str = 'Router'
        GeneralKnowledgeExpert:str = 'GeneralKnowledgeExpert'
        WebSearchExpert:str = 'WebSearchExpert'
        JiraExpert:str = 'JiraExpert'
        QueryXtractionXpert:str = 'QueryXtractionXpert'
        HyDExpert:str = 'HyDExpert'
        RetrieverExpert:str = 'RetrieverExpert'
        RerankingExpert:str = 'RerankingExpert'
        ContextExpert:str = 'ContextExpert'

    def get(xpert:Directory, llm:LLMs, **kwargs):
        if xpert == ExpertFactory.Directory.Router:
            return ExpertRepo.Router.get_router(llm=llm)
        if xpert == ExpertFactory.Directory.GeneralKnowledgeExpert:
            return ExpertRepo.GeneralKnowledgeExpert.get_expert(llm=llm)
        if xpert == ExpertFactory.Directory.WebSearchExpert:
            return ExpertRepo.WebSearchExpert.get_expert(llm=llm)
        if xpert == ExpertFactory.Directory.JiraExpert:
            return ExpertRepo.JiraExpert.get_expert(llm=llm)
        if xpert == ExpertFactory.Directory.QueryXtractionXpert:
            return ExpertRepo.QueryXtractionXpert.get_expert(llm=llm)
        if xpert == ExpertFactory.Directory.HyDExpert:
            return ExpertRepo.HyDExpert.get_expert(llm=llm)
        if xpert == ExpertFactory.Directory.RetrieverExpert:
            return ExpertRepo.RetrieverExpert.get_expert(retriever=kwargs['retriever'])
        if xpert == ExpertFactory.Directory.RerankingExpert:
            return ExpertRepo.RerankingExpert.get_expert(reranker=kwargs['reranker'])
        if xpert == ExpertFactory.Directory.ContextExpert:
            return ExpertRepo.ContextExpert.get_expert(llm=llm)

