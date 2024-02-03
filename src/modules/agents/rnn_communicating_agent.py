import torch
import torch.nn as nn
import torch.nn.functional as F

class MessageGenerationNetwork(nn.Module):
    def __init__(self, input_shape, message_shape, args, n_agents, trajectory_length=10):
        super(MessageGenerationNetwork, self).__init__()
        self.input_shape = input_shape
        self.message_shape = message_shape
        self.args = args
        self.n_agents = n_agents
        self.trajectory_length = trajectory_length

        self.am = AttentionMechanism(self.input_shape, self.message_shape, self.args, self.n_agents, self.trajectory_length)
        self.itgm

    def forward(self, inputs, messages):
        pass

class AttentionMechanism(nn.Module):
    def __init__(self, input_shape, message_shape, args, n_agents, trajectory_length=10):
        super(AttentionMechanism, self).__init__()
        self.input_shape = input_shape
        self.message_shape = message_shape
        self.args = args
        self.n_agents = n_agents
        self.trajectory_length = trajectory_length

        # Not sure if we need to multiply by self.n_agents in the output_dim
        self.qs = nn.Linear(message_shape, self.args.hidden_dim*self.n_agents)
        self.ks = nn.Linear(input_shape*trajectory_length, self.args.hidden_dim*trajectory_length*self.n_agents)
        self.vs = nn.Linear(input_shape*trajectory_length, self.args.hidden_dim*trajectory_length*self.n_agents)


    def forward(self, imagined_trajectory, received_message):
        qs = self.qs(received_message).reshape(self.n_agents, self.args.hidden_dim)
        ks = self.ks(imagined_trajectory).reshape(self.trajectory_length, self.n_agents, self.args.hidden_dim)
        vs = self.vs(imagined_trajectory).reshape(self.trajectory_length, self.n_agents, self.args.hidden_dim)
        sqrt_dk = torch.sqrt(self.args.hidden_dim)

        messages = []
        for i in range(self.n_agents):
            unscaled_alpha = []
            for j in range(self.trajectory_length):
                q = torch.flatten(qs[i])
                unscaled_alpha.append(torch.dot(q, ks[j, i]) / sqrt_dk)
            alpha = torch.softmax(unscaled_alpha)
            message = 0
            for j in range(self.trajectory_length):
                message += alpha[j] * vs[j, i]
            messages.append(message)

        # shape is (n_agents, message_size)
        messages = torch.stack(messages)

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

        self.fa_fc1 = nn.Linear(self.input_shape, self.args.hidden_dim)
        self.fa_fc2 = nn.Linear(self.args.hidden_dim, self.args.n_actions*(self.n_agents-1))

        self.fo_fc1 = nn.Linear(self.args.n_actions*self.n_agents + self.input_shape, self.hidden_dim)
        self.fo_fc2 = nn.Linear(self.hidden_dim, self.input_shape)

        self.pi_fc1 = nn.Linear(self.input_shape+self.message_shape, self.args.hidden_dim)
        self.pi_fc2 = nn.Linear(self.args.hidden_dim, self.args.n_actions)

    def forward(self, imagined_observation, imagined_action, received_message):
        # How to separate out the taus
        fa = F.relu(self.fa_fc1(imagined_observation))
        fa = F.softmax(self.fa_fc2(fa))

        fo = torch.stack([fa, [imagined_action], [imagined_observation]])
        fo = F.relu(self.fo_fc1(fo))
        fo = self.fo_fc1(fo)



class RNNCommunicatingAgent(nn.Module):
    def __init__(self, input_shape, message_shape, args, n_agents):
        super(RNNCommunicatingAgent, self).__init__()
        self.args = args
        self.n_agents = n_agents

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

