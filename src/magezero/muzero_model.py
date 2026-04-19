import torch
from torch import nn
import math
from enum import Enum

"""
MuZero model for Magic: The Gathering.
Consists of:
1. Representation function (h): state -> hidden_state
2. Dynamics function (g): hidden_state, action -> next_hidden_state, reward
3. Prediction function (f): hidden_state -> policy, value

Input and output for AlphaZero compatibility: forward(indices, offsets)
"""

ACTIONS_MAX = 128
GLOBAL_MAX = 2000000

PRIORITY_A_MAX = 128
PRIORITY_B_MAX = 128
CHOOSE_NUM_MAX = 64
MAKE_CHOICE_MAX = 16
TARGETS_MAX = 128
BINARY_MAX = 2
HIDDEN_DIM = 128
EMBEDDING_DIM = 256

class ActionType(Enum):
    PRIORITY = 0,
    CHOOSE_NUM = 1,
    BLANK = 2,
    CHOOSE_TARGET = 3,
    MAKE_CHOICE = 4,
    CHOOSE_USE = 5,


class RepresentationNetwork(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode='sum',
            sparse=False,
            max_norm=1
        )
        self.embedding_dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, indices, offsets):
        emb = self.embedding_bag(indices, offsets)
        emb = self.embedding_dropout(emb)
        hidden_state = self.fc(emb)
        # Normalize hidden state for MuZero stability
        hidden_state = (hidden_state - hidden_state.min(dim=1, keepdim=True)[0]) / (
            hidden_state.max(dim=1, keepdim=True)[0] - hidden_state.min(dim=1, keepdim=True)[0] + 1e-8
        )
        return hidden_state

class PredictionNetwork(nn.Module):
    def __init__(self, hidden_dim, policy_size):
        super().__init__()
        self.player_priority_head = nn.Linear(hidden_dim, policy_size)
        self.opponent_priority_head = nn.Linear(hidden_dim, policy_size)
        self.target_head = nn.Linear(hidden_dim, policy_size)
        self.binary_head = nn.Linear(hidden_dim, 2)
        self.make_choice_head = nn.Linear(hidden_dim, MAKE_CHOICE_MAX)
        self.choose_num_head = nn.Linear(hidden_dim, CHOOSE_NUM_MAX)
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, hidden_state):
        pA = self.player_priority_head(hidden_state)
        pB = self.opponent_priority_head(hidden_state)
        target = self.target_head(hidden_state)
        binary = self.binary_head(hidden_state)
        make_choice = self.make_choice_head(hidden_state)
        choose_num = self.choose_num_head(hidden_state)
        value = self.value_head(hidden_state).squeeze(-1)
        
        policies = {
            ActionType.PRIORITY: pA,
            # We might need a separate ActionType for opponent priority if we want to model it explicitly in MCTS
            ActionType.CHOOSE_TARGET: target,
            ActionType.CHOOSE_USE: binary,
            ActionType.MAKE_CHOICE: make_choice,
            ActionType.CHOOSE_NUM: choose_num
        }
        
        return policies, value

class DynamicsNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        # MuZero dynamics typically takes action as input
        # MTG has complex action space. For now, we embed the action index.
        self.action_embedding = nn.Embedding(action_dim, 32)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh() # Assuming reward is normalized or win/loss signal
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ActionType)),
        )

    def forward(self, hidden_state, action_index):
        # action_index: [batch_size]
        act_emb = self.action_embedding(action_index)
        x = torch.cat([hidden_state, act_emb], dim=1)
        next_hidden_state = self.fc(x)
        action_probs = self.action_head(x)
        action_type_pred = torch.argmax(action_probs, dim=1)  # Predict next action type
        # Normalize next hidden state
        next_hidden_state = (next_hidden_state - next_hidden_state.min(dim=1, keepdim=True)[0]) / (
            next_hidden_state.max(dim=1, keepdim=True)[0] - next_hidden_state.min(dim=1, keepdim=True)[0] + 1e-8
        )
        reward = self.reward_head(next_hidden_state).squeeze(-1)
        return next_hidden_state, reward, action_type_pred

class MuZeroModel(nn.Module):
    def __init__(self, num_embeddings=GLOBAL_MAX, policy_size=PRIORITY_A_MAX):
        super().__init__()
        self.representation = RepresentationNetwork(num_embeddings, EMBEDDING_DIM, HIDDEN_DIM)
        self.prediction = PredictionNetwork(HIDDEN_DIM, policy_size)
        # Max index across all policy heads for dynamics action embedding
        self.dynamics = DynamicsNetwork(HIDDEN_DIM, policy_size)

    def forward(self, indices, offsets):
        """
        AlphaZero-style forward pass: observation -> policy, value
        """
        hidden_state = self.representation(indices, offsets)
        return self.prediction(hidden_state)

    def initial_inference(self, indices, offsets):
        """
        MuZero initial inference: observation -> hidden_state, policy, value
        """
        hidden_state = self.representation(indices, offsets)
        policies, value = self.prediction(hidden_state)
        return hidden_state, policies, value

    def recurrent_inference(self, hidden_state, action_index):
        """
        MuZero recurrent inference: hidden_state, action -> next_hidden_state, reward, policy, value
        """
        next_hidden_state, reward, next_action_type = self.dynamics(hidden_state, action_index)
        policies, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policies, value, next_action_type
