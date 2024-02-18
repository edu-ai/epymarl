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
        self.scheme = scheme
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        if 'action_selector' in vars(args):
            self.action_selector = action_REGISTRY[args.action_selector](args)
        else:
            self.action_selector = None

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
            messages_shape = self.n_agents * self.scheme['message']['vshape']
            input_shape -= messages_shape
            self.agent = agent_REGISTRY[self.args.agent+'_comms'](input_shape, messages_shape, self.args, self.n_agents)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        if self.allow_communications:
            # These are duplicated here to provide the t-1 input to generate messages used for current input
            input_shape += scheme["obs"]["vshape"]
            input_shape += scheme["actions_onehot"]["vshape"][0]
            if self.args.obs_agent_id:
                input_shape += self.n_agents
            
            # Communication flag
            # if self.myopic_communications:
            #     input_shape += 1
            input_shape += scheme['message']['vshape'] * self.n_agents

        return input_shape

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, max(0, t-1)]))
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        if self.allow_communications:
            # This chunk builds previous observation
            inputs.append(batch["obs"][:, max(0, t-1)])
            if self.args.obs_last_action:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, max(0, t-2)]))
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            
            inputs.append(th.zeros_like(batch["actions_onehot"][:, max(0, t-1)]))
            
            inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
            
            # We take the message from 2 timesteps ago, the agent will use that to generate the message from the 
            # previous timestep, and that is what we use to inform the current action. This is to fix backprop issues
            messages = batch['message'][:, max(0, t-2)]
            messages = messages.reshape(bs, -1)
            messages = th.cat([messages]*self.n_agents, dim=0)
            messages = messages.reshape(bs, self.n_agents, -1)

            inputs = inputs.reshape(bs, self.n_agents, -1)
            inputs = th.cat([inputs, messages], dim=-1)
            inputs = inputs.reshape(bs*self.n_agents, -1)
        return inputs


