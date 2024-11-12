from MoE.base.expert import Expert
from MoE.base.mixture import MoE
from MoE.xperts.expert_factory import ExpertFactory


class GenChatMoE(MoE):
    def define_xpert_impl(self, state: MoE.State, xpert: Expert) -> dict:
        if xpert.name == ExpertFactory.Directory.GeneralKnowledgeExpert:
            return self.run_gen_xpert(state=state, xpert=xpert)
        if xpert.name == ExpertFactory.Directory.WebSearchExpert:
            return self.run_web_xpert(state=state, xpert=xpert)
    
    def run_gen_xpert(self, state: MoE.State, xpert: Expert) -> dict:
        output = xpert.invoke({
            'input': state['expert_input'],
            'chat_history': state['ephemeral_mem'].messages[-5:],
        })

        state['next'] = self.router.name
        state['expert_output'] = output
        return state
    
    def run_web_xpert(self, state: MoE.State, xpert: Expert) -> dict:
        output = xpert.invoke({
            'input': state['expert_input'],
            'chat_history': state['ephemeral_mem'].messages[-5:]
        })

        state['next'] = self.router.name
        state['expert_output'] = output
        return state