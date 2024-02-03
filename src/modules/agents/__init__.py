REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_communicating_agent import RNNCommunicatingAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_comms"] = RNNCommunicatingAgent