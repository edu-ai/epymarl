from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

class Controller:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.allow_communications = args.allow_communications
        self.myopic_communications = args.myopic_communications
        self.intention_sharing = args.intention_sharing

        if self.allow_communications:
            input_shape, message_shape = self._get_input_shape(scheme)
            self._build_agents(input_shape, message_shape)
        else:
            input_shape = self._get_input_shape(scheme)
            self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape, message_shape=None):
        if not self.allow_communications:
            self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        else:
            # self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
            self.agent = agent_REGISTRY[self.args.agent+'_comms'](input_shape, message_shape, self.args, self.n_agents)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        if self.allow_communications:
            if self.myopic_communications:
                input_shape += 1
            return input_shape, scheme['message']['vshape']
        return input_shape

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        print('message shape:', batch['message'][:, t].shape, '\t|\tbatch size:', bs, '\t|\tt:', t, '\t|\tn:', self.n_agents, '\t|\tn_actions:', self.args.n_actions)
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)


        if self.allow_communications:
            messages = batch['message'][:, t]
            messages = messages.reshape(bs, self.n_agents, -1)
            inputs = inputs.reshape(bs, self.n_agents, -1)
            print(messages.shape, inputs.shape)
            inputs = th.cat([inputs, messages], dim=-1)
            print(inputs.shape)
            inputs = inputs.reshape(bs*self.n_agents, -1)

        return inputs


