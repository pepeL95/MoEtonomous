from MoE.base.expert import Expert
from MoE.base.mixture import MoE
from MoE.xperts.expert_factory import ExpertFactory

class ListPalMoE(MoE):
    def define_xpert_impl(self, state: MoE.State, xpert: Expert) -> dict:
        if xpert.name == ExpertFactory.Directory.JiraExpert:
            return self._run_jira_xpert(state, xpert)

    
    def _run_jira_xpert(self, state: MoE.State, xpert: Expert) -> dict:
        output = xpert.invoke({'input': state['expert_input'], 'chat_history': state['ephemeral_mem'].messages[-5:]})
        
        state['expert_output'] = output
        state['next'] = self.router.name
        return state