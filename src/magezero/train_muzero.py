import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple

from magezero.muzero_model import MuZeroModel, ActionType
from magezero.muzero_mcts import MuZeroMCTS
from magezero.dataset import H5Indexed, collate_batch

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class MuZeroTrainer:
    def __init__(self, model: MuZeroMCTS, rollout_steps = 5, lr=1e-3, weight_decay=1e-4):
        self.mcts_model = model
        self.mcts_model.model = self.mcts_model.model.to(DEVICE)
        self.optimizer = optim.AdamW(self.mcts_model.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler() # For mixed precision training

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.rollout_steps = rollout_steps

    def train_step(self, batch):
        """
        One training step using a batch of transitions.
        Each element in batch is (state_t, action_t, next_state_t, reward_t, policy_t, value_t)
        """
        self.mcts_model.model.train()
        self.optimizer.zero_grad()
        idxs, offsets, target_policies, target_values, is_players, action_types, rollout_rewards, rollout_action_types = batch
        idxs = idxs.to(DEVICE)
        offsets = offsets.to(DEVICE)
        target_values = target_values.to(DEVICE).squeeze(-1)
        rollout_loss = 0.0
        ## Initial Step 
        hidden_state = self.mcts_model.model.representation(idxs, offsets)
        curr_hidden_state = hidden_state
        for idx in range(self.rollout_steps):
            policies, values = self.mcts_model.model.prediction(hidden_state)
            loss_value = self.mse_loss(values, target_values)
            # loss_policy = self.ce_loss(pA, target_policies) 
            # 3. Dynamics pass (s_t, a_t -> pred_s_{t+1}, pred_r_t)
            # We'll take the most visited action as the 'action taken'.
            for batch_idx in range(hidden_state.shape[0]):
                action_type = action_types[batch_idx].item()
                hidden_state_root_node = hidden_state[batch_idx]
                batch_policies = {k:x.unsqueeze(0) for k,x in policies[batch_idx].item()}
                policies, policy_mask, action_k, value_k = \
                    self.mcts_model.search(curr_hidden_state[batch_idx].unsqueeze(0), policies, action_type) 

            pred_next_hidden, pred_reward, next_action_type = self.mcts_model.model.dynamics(hidden_state, action_k)
            
           

        total_loss = loss_policy + loss_value 

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), loss_policy.item(), loss_value.item(), loss_consistency.item()

def train(deck: str, version: int, epochs: int = 10, batch_size: int = 64):
    inner_model = MuZeroModel()
    model = MuZeroMCTS(inner_model)
    trainer = MuZeroTrainer(model)
    
    data_dir = f"data/{deck}/ver{version}/training"
    dataset = H5Indexed(data_dir, rollout=True, rollout_len=trainer.rollout_steps)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    breakpoint()

    print(f"Starting MuZero training for {deck} ver{version}...")
    for epoch in range(epochs):
        epoch_loss = 0
        t0 = time.time()
        for i, batch in enumerate(loader):
            loss, lp, lv, lc = trainer.train_step(batch)
            epoch_loss += loss
            if i % 100 == 0:
                print(f"Batch {i}, Loss: {loss:.4f} (Policy: {lp:.4f}, Value: {lv:.4f}, Consistency: {lc:.4f})")
        
        avg_loss = epoch_loss / len(loader)
        dt = time.time() - t0
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}, Time: {dt:.1f}s")

        # Save model
        save_dir = f"models/{deck}/ver{version+1}_muzero"
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
        }, f"{save_dir}/model.pt")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    train(args.deck, args.version)

if __name__ == "__main__":
    main()