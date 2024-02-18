import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.agents.rnn_communicating_agent import RNNCommunicatingAgent

class RNNNSCommunicatingAgent(nn.Module):
    def __init__(self, input_shape, messages_shape, args, n_agents, trajectory_length=10):
        super(RNNNSCommunicatingAgent, self).__init__()
        self.args = args
        self.n_agents = n_agents
        self.trajectory_length = trajectory_length

        self.obs_shape = input_shape // 2
        self.messages_shape = messages_shape
        self.agents = th.nn.ModuleList([RNNCommunicatingAgent(input_shape, messages_shape, args, n_agents, trajectory_length) for _ in range(self.n_agents)])

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                qs.append(q)
            return th.cat(qs), th.cat(hiddens).unsqueeze(0)
        else:
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, inputs.shape[-1])
                q, h = self.agents[i](inputs[:, i], hidden_state[:, i])
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
            return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)

