import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionMechanism(nn.Module):
    def __init__(self, obs_shape, messages_shape, args, n_agents, trajectory_length=3):
        super(AttentionMechanism, self).__init__()
        self.tau_shape = obs_shape + args.n_actions
        self.messages_shape = messages_shape
        self.args = args
        self.n_agents = n_agents
        self.trajectory_length = trajectory_length

        self.qs = nn.Linear(messages_shape, self.args.hidden_dim)
        self.ks = nn.Linear(self.tau_shape, self.args.hidden_dim)
        self.vs = nn.Linear(self.tau_shape, self.messages_shape // self.n_agents)


    def forward(self, imagined_trajectory, received_messages):
        # need to duplicate messages for every agent to see
        qs = self.qs(received_messages)
        taus = imagined_trajectory.reshape(-1, self.tau_shape)
        k = self.ks(taus)
        v = self.vs(taus)
        sqrt_dk = math.sqrt(self.args.hidden_dim)
        messages = []
        for i in range(received_messages.shape[0]):
            unscaled_alpha = []
            for j in range(self.trajectory_length):
                if len(k) > self.trajectory_length:
                    unscaled_alpha.append(th.dot(qs[i], k[j*self.n_agents+i, :]) / sqrt_dk)
                else:
                    unscaled_alpha.append(th.dot(qs[i], k[j+i, :]) / sqrt_dk)
            unscaled_alpha = th.Tensor(unscaled_alpha)
            alpha = th.softmax(unscaled_alpha, -1)
            if len(k) > self.trajectory_length:
                message = sum(alpha[j]*v[j*self.n_agents+i, :] for j in range(self.trajectory_length))
            else:
                message = sum(alpha[j]*v[j+i, :] for j in range(self.trajectory_length))
            messages.append(message)

        # shape is (n_agents, message_size)
        messages = th.stack(messages)

        return messages

class ImaginedTrajectory(nn.Module):
    def __init__(self, obs_shape, messages_shape, args, n_agents, pi, trajectory_length=3):
        super(ImaginedTrajectory, self).__init__()
        self.args = args
        self.obs_shape = obs_shape
        self.messages_shape = messages_shape
        self.n_agents = n_agents
        self.trajectory_length = trajectory_length
        self.pi = pi
        # Define fa, fo and pi^i

        self.fa_fc1 = nn.Linear(self.obs_shape, self.args.hidden_dim)
        self.fa_fc2 = nn.Linear(self.args.hidden_dim, self.args.n_actions*(self.n_agents-1))

        self.fo_fc1 = nn.Linear(self.args.n_actions*self.n_agents + self.obs_shape, self.args.hidden_dim)
        self.fo_fc2 = nn.Linear(self.args.hidden_dim, self.obs_shape)

        if self.args.use_rnn:
            self.fo_rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
            self.fa_rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.fo_rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.fa_rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
    
    def forward(self, message, obs, action, h_fa, h_fo, h_pi):
        observations = [obs]
        actions = [action]
        for i in range(self.trajectory_length-1):
            obs, action, h_fa, h_fo, h_pi = self.forward_helper(obs, action, message, h_fa, h_fo, h_pi)
            observations.append(obs)
            actions.append(action)
        taus = []

        for i in range(self.trajectory_length):
            tau = th.cat([observations[i], actions[i]], -1)
            taus.append(tau)
        taus = th.stack(taus) # (trajectory_length, n_agents, vs)

        return taus, h_fa, h_fo, h_pi

    def forward_helper(self, imagined_observation, imagined_action, received_message, hidden_state_fa, hidden_state_fo, hidden_state_pi):
        # How to separate out the taus
        fa, next_hidden_state_fa = self.forward_fa(imagined_observation, hidden_state_fa)
        fo, next_hidden_state_fo = self.forward_fo(fa, imagined_action, imagined_observation, hidden_state_fo)
        pi, next_hidden_state_pi = self.forward_pi(imagined_observation, received_message, hidden_state_pi)

        return fo, pi, next_hidden_state_fa, next_hidden_state_fo, next_hidden_state_pi
    
    def forward_fo(self, fa, imagined_action, imagined_observation, hidden_state):
        x_in = th.cat([fa, imagined_action, imagined_observation], -1)
        x = F.relu(self.fo_fc1(x_in))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.fo_rnn(x, h_in)
        else:
            h = F.relu(self.fo_rnn(x))
        q = self.fo_fc2(h)

        return q, h
    
    def forward_fa(self, obs, hidden_state):
        x = F.relu(self.fa_fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.fa_rnn(x, h_in)
        else:
            h = F.relu(self.fa_rnn(x))
        q = self.fa_fc2(h)

        q_original_shape = q.shape
        q_size = 1
        for s in q.shape:
            q_size *= s
        n_actions = self.args.n_actions
        q = q.flatten().reshape(q_size//(n_actions*(self.n_agents-1)), n_actions*(self.n_agents-1))

        qs_out = []
        for i in range(self.n_agents-1):
            q_out = th.softmax(q[:, i*n_actions:(i+1)*n_actions], -1)
            qs_out.append(q_out)
        q = th.stack(qs_out).reshape(q_original_shape)

        return q, h

    def forward_pi(self, obs, received_message, hidden_state):
        q, h = self.pi(obs.detach(), received_message.detach(), hidden_state.detach())
        return q, h

class Pi(nn.Module):
    def __init__(self, obs_shape, messages_shape, args, n_agents, trajectory_length=10):
        super(Pi, self).__init__()
        self.args = args
        self.obs_shape = obs_shape
        self.messages_shape = messages_shape
        self.n_agents = n_agents

        self.pi_fc1 = nn.Linear(self.obs_shape+self.messages_shape, self.args.hidden_dim)
        self.pi_fc2 = nn.Linear(self.args.hidden_dim, self.args.n_actions)
        if self.args.use_rnn:
            self.pi_rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.pi_rnn = nn.Linear(args.hidden_dim, args.hidden_dim)

    def init_hidden(self):
        return self.pi_fc1.weight.new(1, self.args.hidden_dim).zero_()
    
    def forward(self, obs, received_message, hidden_state):
        combined_inputs = th.cat([obs, received_message], -1)
        x = F.relu(self.pi_fc1(combined_inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.pi_rnn(x, h_in)
        else:
            h = F.relu(self.pi_rnn(x))
        q = self.pi_fc2(h)
        return q, h

class RNNCommunicatingAgent(nn.Module):
    def __init__(self, input_shape, messages_shape, args, n_agents):
        super(RNNCommunicatingAgent, self).__init__()
        self.args = args
        self.n_agents = n_agents
        self.trajectory_length = args.trajectory_length

        self.obs_shape = (input_shape - args.n_actions) // 2
        self.messages_shape = messages_shape

        self.pi = Pi(self.obs_shape, messages_shape, args, n_agents)
        self.itgm = ImaginedTrajectory(self.obs_shape, messages_shape, args, n_agents, self.pi, args.trajectory_length)
        self.am = AttentionMechanism(self.obs_shape, messages_shape, args, n_agents, args.trajectory_length)

    def init_hidden(self):
        # make hidden states on same device as model
        h_fa = self.itgm.fa_fc1.weight.new(1, self.args.hidden_dim).zero_()
        h_fo = self.itgm.fo_fc1.weight.new(1, self.args.hidden_dim).zero_()
        h_pi = self.pi.init_hidden()
        return th.cat([h_fa, h_fo, h_pi], -1)
    
    def forward(self, inputs, hidden_state):
        h_fa = hidden_state[..., :self.args.hidden_dim]
        h_fo = hidden_state[..., self.args.hidden_dim:2*self.args.hidden_dim]
        h_pi = hidden_state[..., -self.args.hidden_dim:]

        observation = inputs[..., :self.obs_shape]
        prev_observation = inputs[..., self.obs_shape:self.obs_shape*2]
        prev_action = inputs[..., self.obs_shape*2:self.obs_shape*2+self.args.n_actions].detach()
        prev_message = inputs[..., -self.messages_shape:]

        taus, h_fa, h_fo, h_pi_old = self.itgm.forward(
            prev_message, prev_observation, prev_action, h_fa, h_fo, h_pi
        )

        am_inputs = taus.reshape((prev_message.shape[0], -1))
        new_message = self.am.forward(am_inputs, prev_message)
        pi_input_message = th.cat([new_message]*self.n_agents, 0).reshape(new_message.shape[0], -1)

        action, h_pi = self.pi(observation, pi_input_message, h_pi)
        hidden_state = th.cat([h_fa, h_fo, h_pi], -1)
        try:
            q = th.cat([new_message, action], -1)
        except:
            print("Error")
            print(f"Inputs shape: {inputs.shape}\t|\tTaus shape: {taus.shape}\t|\tMessage shape: {new_message.shape}")
            print(f"new_message_shape: {new_message.shape}\t|\ttau_shape: {taus[1, ..., -self.args.n_actions:].shape}")
        return q, hidden_state

