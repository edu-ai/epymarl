import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionMechanism(nn.Module):
    def __init__(self, input_shape, message_shape, args, n_agents, trajectory_length=10):
        super(AttentionMechanism, self).__init__()
        self.tau_shape = input_shape
        self.message_shape = message_shape
        self.args = args
        self.n_agents = n_agents
        self.trajectory_length = trajectory_length

        self.qs = nn.Linear(message_shape, self.args.hidden_dim)
        self.ks = nn.Linear(self.tau_shape, self.args.hidden_dim)
        self.vs = nn.Linear(self.tau_shape, self.message_shape // self.n_agents)


    def forward(self, imagined_trajectory, received_messages):
        # need to duplicate messages for every agent to see
        qs = self.qs(received_messages)
        ks, vs = [], []
        for i in range(self.trajectory_length):
            tau = imagined_trajectory[..., i*self.tau_shape:(i+1)*self.tau_shape]
            ks.append(self.ks(tau))
            vs.append(self.vs(tau))
        sqrt_dk = math.sqrt(self.args.hidden_dim)
        messages = []
        for i in range(received_messages.shape[0]):
            unscaled_alpha = []
            for j in range(self.trajectory_length):
                try:
                    unscaled_alpha.append(th.dot(qs[i], ks[j][i]) / sqrt_dk)
                except:
                    print(f"received_messages shape: {received_messages.shape}\t|\ttau shape: {tau.shape}\t|\tks[j][i] shape: {ks[j].shape}\t|\tqs shape: {qs.shape}")
            unscaled_alpha = th.Tensor(unscaled_alpha)
            alpha = th.softmax(unscaled_alpha, -1)
            message = 0
            for j in range(self.trajectory_length):
                a = alpha[j]
                v = vs[j][i]
                message += a * v
            messages.append(message)

        # shape is (n_agents, message_size)
        messages = th.stack(messages)

        return messages

class ImaginedTrajectory(nn.Module):
    def __init__(self, input_shape, message_shape, args, n_agents, trajectory_length=10):
        super(ImaginedTrajectory, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.message_shape = message_shape
        self.n_agents = n_agents
        self.trajectory_length = trajectory_length
        # Define fa, fo and pi^i

        self.fa_fc1 = nn.Linear(self.input_shape - self.args.n_actions, self.args.hidden_dim)
        self.fa_fc2 = nn.Linear(self.args.hidden_dim, self.args.n_actions*(self.n_agents-1))

        self.fo_fc1 = nn.Linear(self.args.n_actions*(self.n_agents - 1) + self.input_shape, self.args.hidden_dim)
        self.fo_fc2 = nn.Linear(self.args.hidden_dim, self.input_shape-self.args.n_actions)

        self.pi_fc1 = nn.Linear(self.input_shape+self.message_shape-self.args.n_actions, self.args.hidden_dim)
        self.pi_fc2 = nn.Linear(self.args.hidden_dim, self.args.n_actions)
        if self.args.use_rnn:
            self.pi_rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
            self.fo_rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
            self.fa_rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.pi_rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.fo_rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.fa_rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
    
    def forward(self, inputs, h_fa, h_fo, h_pi):
        obs = inputs[..., :self.input_shape-self.args.n_actions]
        pi = inputs[..., self.input_shape-self.args.n_actions:-self.message_shape]
        message = inputs[..., -self.message_shape:]

        observations = [obs]
        actions = [pi]
        hs = [h_fa, h_fo, h_pi]
        for i in range(self.trajectory_length-1):
            _, obs, pi, h_fa, h_fo, h_pi = self.forward_helper(obs, pi, message, h_fa, h_fo, h_pi)
            hs.append(h_fa)
            hs.append(h_fo)
            hs.append(h_pi)
            observations.append(obs)
            actions.append(pi)
        taus = []

        for i in range(self.trajectory_length):
            tau = th.cat([observations[i], actions[i]], -1)
            taus.append(tau)
        taus = th.stack(taus) # (trajectory_length, n_agents, vs)

        return message, taus, h_fa, h_fo, h_pi

    def forward_helper(self, imagined_observation, imagined_action, received_message, hidden_state_fa, hidden_state_fo, hidden_state_pi):
        # How to separate out the taus
        fa, next_hidden_state_fa = self.forward_fa(imagined_observation, hidden_state_fa)
        fo, next_hidden_state_fo = self.forward_fo(fa, imagined_action, imagined_observation, hidden_state_fo)
        pi, next_hidden_state_pi = self.forward_pi(imagined_observation, received_message, hidden_state_pi)

        return received_message, fo, pi, next_hidden_state_fa, next_hidden_state_fo, next_hidden_state_pi
    
    def forward_fo(self, fa, imagined_action, imagined_observation, hidden_state):
        x = th.cat([fa, imagined_action, imagined_observation], -1)
        x = F.relu(self.fo_fc1(x))
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
    def __init__(self, input_shape, message_shape, args, n_agents, trajectory_length=10):
        super(RNNCommunicatingAgent, self).__init__()
        self.args = args
        self.n_agents = n_agents
        self.trajectory_length = trajectory_length

        self.input_shape = input_shape
        self.message_shape = message_shape

        self.itgm = ImaginedTrajectory(input_shape, message_shape, args, n_agents, trajectory_length)
        self.am = AttentionMechanism(input_shape, message_shape, args, n_agents, trajectory_length)

    def init_hidden(self):
        # make hidden states on same device as model
        h_fa = self.itgm.fa_fc1.weight.new(1, self.args.hidden_dim).zero_()
        h_fo = self.itgm.fo_fc1.weight.new(1, self.args.hidden_dim).zero_()
        h_pi = self.itgm.pi_fc1.weight.new(1, self.args.hidden_dim).zero_()
        return th.cat([h_fa, h_fo, h_pi], -1)
    
    def forward(self, inputs, hidden_state):
        h_fa = hidden_state[..., :self.args.hidden_dim]
        h_fo = hidden_state[..., self.args.hidden_dim:2*self.args.hidden_dim]
        h_pi = hidden_state[..., -self.args.hidden_dim:]

        message, taus, h_fa, h_fo, h_pi = self.itgm.forward(inputs, h_fa, h_fo, h_pi)
        new_message = self.am.forward(taus.reshape((message.shape[0], -1)), message)

        hidden_state = th.cat([h_fa, h_fo, h_pi], -1)
        try:
            q = th.cat([new_message, taus[1, ..., -self.args.n_actions:]], -1)
        except:
            print("Error")
            print(f"Inputs shape: {inputs.shape}\t|\tTaus shape: {taus.shape}\t|\tMessage shape: {message.shape}")
            print(f"new_message_shape: {new_message.shape}\t|\ttau_shape: {taus[1, ..., -self.args.n_actions:].shape}")
        return q, hidden_state

