from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
from .controller import Controller



def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs

def sample_gumbel(shape, eps=1e-20, tens_type=th.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -th.log(-th.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=-1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


# This multi-agent controller shares parameters between agents
class MADDPGMAC(Controller):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        # Only select actions for the selected batch elements in bs
        if self.allow_communications:
            agent_outputs, message = self.forward(ep_batch, t_ep)
        else:
            agent_outputs = self.forward(ep_batch, t_ep)
        chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1)
        
        if self.allow_communications:
            return chosen_actions, message
        else:
            return chosen_actions

    def target_actions(self, ep_batch, t_ep):
        agent_outputs, messages = self.forward(ep_batch, t_ep)
        return onehot_from_logits(agent_outputs)

    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        if self.allow_communications:
            messages = agent_outs[:, :self.scheme['message']['vshape']]
            agent_outs = agent_outs[:, -self.args.n_actions:]
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions==0] = -1e10
        if self.allow_communications:
            return agent_outs, messages.view(ep_batch.batch_size, self.n_agents, -1).detach()
        else:
            return agent_outs

    def init_hidden_one_agent(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1)  # bav

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
