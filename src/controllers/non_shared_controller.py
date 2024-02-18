from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .controller import Controller
import torch as th


class NonSharedMAC(Controller):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if self.allow_communications:
            agent_outputs, message = self.forward(ep_batch, t_ep, test_mode=test_mode)
        else:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        
        if self.allow_communications:
            return chosen_actions, message
        else:
            return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        if self.allow_communications:
            message = agent_outs[:, :self.scheme['message']['vshape']]
            agent_outs = agent_outs[:, -self.args.n_actions:]

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        if self.allow_communications:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), message.view(ep_batch.batch_size, self.n_agents, -1).detach()
        else:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)