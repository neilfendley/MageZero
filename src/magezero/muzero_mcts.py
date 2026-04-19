import math
import torch
import numpy as np
from torch import nn
from typing import Dict, List, Optional, Tuple, Any
from magezero.muzero_model import (
    ActionType, MuZeroModel, PRIORITY_A_MAX, PRIORITY_B_MAX, TARGETS_MAX, BINARY_MAX
)

class MuZeroNode:
    def __init__(self, prior: float, action_type: ActionType = ActionType.PRIORITY):
        self.prior = prior
        self.action_type = action_type
        self.visit_count = 0
        self.value_sum = 0
        self.children: Dict[int, MuZeroNode] = {}
        self.hidden_state: Optional[torch.Tensor] = None
        self.reward = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_type: ActionType, priors: torch.Tensor, hidden_state: torch.Tensor, reward: float):
        self.hidden_state = hidden_state
        self.reward = reward
        self.action_type = action_type
        
        # Determine number of actions based on type
        if action_type == ActionType.CHOOSE_USE:
            num_actions = 2
        else:
            num_actions = priors.shape[0]

        for action in range(num_actions):
            # For recurrent steps, we don't know the NEXT action type yet.
            # We default to PRIORITY but this should ideally be predicted.
            self.children[action] = MuZeroNode(priors[action].item(), ActionType.PRIORITY)

    def is_expanded(self) -> bool:
        return len(self.children) > 0

class MuZeroMCTS:
    def __init__(self, model: MuZeroModel, config: Optional[Dict] = None):
        self.model = model
        self.config = {
            "num_simulations": 100,
            "pb_c_base": 19652,
            "pb_c_init": 1.25,
            "discount": 0.99,
            "root_dirichlet_alpha": 0.3,
            "root_exploration_fraction": 0.25,
        }
        if config:
            self.config.update(config)

    def get_policy_head(self, action_type: ActionType, policies: Dict[ActionType, torch.Tensor]) -> torch.Tensor:
        return policies[action_type][0]

    # def search(self, indices: torch.Tensor, offsets: torch.Tensor, action_type: ActionType = ActionType.PRIORITY) -> Tuple[int, Dict[int, float], float]:
    def search(self, hidden_state_root_node: torch.Tensor, prior_policies: Dict[ActionType, torch.Tensor], action_type: ActionType = ActionType.PRIORITY) -> Tuple[Dict[ActionType, torch.Tensor], int, float]:
        """
        Perform MuZero MCTS search from the given observation and action type.
        """
        root = MuZeroNode(0, action_type)
        
        # Get correct priors for the root action type
        priors = prior_policies[action_type]
        # Note: should softmax be applied here?
        priors = torch.softmax(priors, dim=0)
        self.add_exploration_noise(root, priors)
        root.expand(action_type, priors, hidden_state_root_node, 0.0)

        for _ in range(self.config["num_simulations"]):
            node = root
            search_path = [node]
            history = []

            # 1. Selection
            while node.is_expanded():
                action, node = self.select_child(node)
                search_path.append(node)
                history.append(action)

            # 2. Expansion and Evaluation
            parent = search_path[-2]
            action = history[-1]
            
            with torch.no_grad():
                next_hidden_state, reward, policies, value, predicted_next_action_type_idx = self.model.recurrent_inference(
                    parent.hidden_state, torch.tensor([action], device=parent.hidden_state.device)
                )
            
            # Map predicted index back to ActionType
            next_type = list(ActionType)[predicted_next_action_type_idx.item()]
            
            priors_raw = policies[next_type]
            priors = torch.softmax(priors_raw[0], dim=0)
            
            node.expand(next_type, priors, next_hidden_state, reward.item())
            
            # 3. Backpropagation
            self.backpropagate(search_path, value.item())

        # Select action based on visit counts
        action = self.select_final_action(root)
        
        # Collect policies (visit distributions) for all types encountered
        # In a real game, we only care about the root action
        # but for training we might want the policy for the action we're taking.
        final_policies = {}
        node = root
        # Simplified logic to extract policy from visit counts
        for atype in ActionType:
            if atype == root.action_type:
                total_visits = sum(child.visit_count for child in root.children.values())
                policy = torch.zeros(len(root.children))
                for a, child in root.children.items():
                    policy[a] = child.visit_count / total_visits if total_visits > 0 else 0
                final_policies[atype] = policy
            else:
                # Default empty policy for other types
                final_policies[atype] = torch.zeros(1) # Or appropriate size

        return final_policies, action, root.value

    def get_policy_from_node(self, node: MuZeroNode):
        visit_counts = [child.visit_count for a, child in node.children.items()]
        return visit_counts


    def select_child(self, node: MuZeroNode) -> Tuple[int, MuZeroNode]:
        best_score = -float("inf")
        best_action = -1
        best_child = None

        total_visit_count = sum(child.visit_count for child in node.children.values())

        for action, child in node.children.items():
            score = self.ucb_score(node, child, total_visit_count)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, parent: MuZeroNode, child: MuZeroNode, total_visit_count: int) -> float:
        pb_c = math.log((total_visit_count + self.config["pb_c_base"] + 1) / self.config["pb_c_base"]) + self.config["pb_c_init"]
        pb_c *= math.sqrt(total_visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value if child.visit_count > 0 else 0
        return value_score + prior_score

    def backpropagate(self, search_path: List[MuZeroNode], value: float):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.config["discount"] * value

    def add_exploration_noise(self, node: MuZeroNode, priors: torch.Tensor):
        pass

    def select_final_action(self, root: MuZeroNode) -> int:
        best_visit_count = -1
        best_action = -1
        for action, child in root.children.items():
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_action = action
        return best_action

# class MuZeroMCTSModel(nn.Module):
#     """
#     Wraps MuZeroModel and MuZeroMCTS to provide a model-like interface
#     that performs MCTS search during its forward pass.
#     """
#     def __init__(self, model: MuZeroModel, config: Optional[Dict] = None):
#         super().__init__()
#         self.model = model
#         self.mcts = MuZeroMCTS(model, config)

#     def forward(self, indices: torch.Tensor, offsets: torch.Tensor, action_type: ActionType = ActionType.PRIORITY):
#         batch_size = offsets.shape[0]
#         device = indices.device
        
#         # Initialize output tensors
#         pA = torch.zeros((batch_size, PRIORITY_A_MAX), device=device)
#         pB = torch.zeros((batch_size, PRIORITY_B_MAX), device=device)
#         target = torch.zeros((batch_size, TARGETS_MAX), device=device)
#         binary = torch.zeros((batch_size, BINARY_MAX), device=device)
#         value = torch.zeros((batch_size,), device=device)
        
#         for i in range(batch_size):
#             # Slice batch for single MCTS search
#             start_idx = offsets[i].item()
#             end_idx = offsets[i+1].item() if i+1 < batch_size else indices.shape[0]
            
#             sub_indices = indices[start_idx:end_idx]
#             sub_offsets = torch.tensor([0], device=device)
            
#             # Perform MCTS search
#             _, visit_counts, root_value = self.mcts.search(sub_indices, sub_offsets, action_type)
            
#             # Update value
#             value[i] = root_value
            
#             # Update visit count policy for the relevant head
#             total_visits = sum(visit_counts.values())
#             if total_visits > 0:
#                 if action_type == ActionType.PRIORITY:
#                     for a, count in visit_counts.items():
#                         if a < PRIORITY_A_MAX:
#                             pA[i, a] = count / total_visits
#                 elif action_type == ActionType.CHOOSE_TARGET:
#                     for a, count in visit_counts.items():
#                         if a < TARGETS_MAX:
#                             target[i, a] = count / total_visits
#                 elif action_type == ActionType.CHOOSE_USE:
#                     for a, count in visit_counts.items():
#                         if a < BINARY_MAX:
#                             binary[i, a] = count / total_visits
                            
#         return pA, pB, target, binary, value
