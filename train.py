import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Subset
from dataset import LabeledStateDataset, collate_batch, load_dataset_from_directory
from typing import Set, List, Tuple
ACTIONS_MAX = 128
GLOBAL_MAX = 100000
EPOCH_COUNT = 20


class Net(nn.Module):
    def __init__(self, num_embeddings, policy_size_A):
        super().__init__()

        embedding_dim = 512  # Output of EmbeddingBag, matches your original fc1 output
        hidden_dim_mlp = 256  # Output of the main MLP block, matches your original fc2 output

        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode='sum',
            sparse=True
        )
        # 1. Define the learnable bias parameter
        self.embedding_bias = nn.Parameter(torch.zeros(embedding_dim))
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dropout = nn.Dropout(p=0.3)

        self.fc_after_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp),  # From 512 to 256
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim_mlp, policy_size_A)  # From 256 to A
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim_mlp, 1),  # From 256 to 1
            nn.Tanh()
        )

    def forward(self, indices, offsets):
        emb = self.embedding_bag(indices, offsets)
        #emb = self.embedding_dropout(emb)
        emb = self.embedding_norm(emb + self.embedding_bias)
        h = self.fc_after_embedding(emb)
        return self.policy_head(h), self.value_head(h).squeeze(-1)



def train():

    combined_ds = load_dataset_from_directory("data/UWTempo/ver4/training")
    #combined_ds = LabeledStateDataset("data/UWTempo/ver3/training/training.bin")
    dl = DataLoader(combined_ds, batch_size=128, shuffle=True, num_workers=16, collate_fn=collate_batch, pin_memory=True, persistent_workers=True)
    model = Net(GLOBAL_MAX, ACTIONS_MAX).cuda()


    sparse_params = model.embedding_bag.parameters()
    opt_sparse = optim.SparseAdam(sparse_params, lr=5e-4)#optim.SparseAdam(sparse_params, lr=1e-3)

    # Optimizer for all other (dense) parameters
    dense_params = []
    for name, param in model.named_parameters():
        if "embedding_bag" not in name:  # Exclude embedding_bag parameters
            dense_params.append(param)
    opt_dense = optim.Adam(dense_params, lr=5e-4)

    checkpoint_path = f"models/model0/ckpt_11.pt"  # Example: Starting from ckpt_16

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_sparse.load_state_dict(checkpoint['optimizer_sparse_state_dict'])
        opt_dense.load_state_dict(checkpoint['optimizer_dense_state_dict'])
        # start_epoch = checkpoint['epoch'] + 1 # Use this if you want to continue the same run
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        # If continuing a run, you might load the epoch and loss too:
        # last_epoch = checkpoint['epoch']
        # print(f"Resuming training from epoch {last_epoch + 1}")

    except FileNotFoundError:
        print(f"INFO: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
    except Exception as e:
        print(f"ERROR: Could not load checkpoint. {e}. Starting from scratch.")

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()


    for epoch in range(1, EPOCH_COUNT+1):
        total_p_loss, total_v_loss = 0.0, 0.0  # Initialize as floats
        model.train()

        # 4. Update the loop to unpack all parts returned by collate_batch
        for batch_indices, batch_offsets, batch_policy_labels, batch_value_labels in dl:
            # 5. Move new input tensors to CUDA
            batch_indices = batch_indices.cuda()
            batch_offsets = batch_offsets.cuda()
            batch_policy_labels = batch_policy_labels.cuda()
            batch_value_labels = batch_value_labels.cuda()

            # 6. Model call uses indices and offsets
            policy_logits, value_pred = model(batch_indices, batch_offsets)

            policy_target_indices = torch.argmax(batch_policy_labels, dim=1)
            lp = ce(policy_logits, batch_policy_labels)

            lv = mse(value_pred, batch_value_labels.squeeze(-1))  # Ensure batch_value_labels is shape [batch_size]
            loss = lp + lv

            opt_sparse.zero_grad()
            opt_dense.zero_grad()
            loss.backward()
            opt_sparse.step()
            opt_dense.step()

            total_p_loss += lp.item()
            total_v_loss += lv.item()

        avg_p_loss = total_p_loss / len(dl)
        avg_v_loss = total_v_loss / len(dl)
        print(f"Epoch {epoch}  policy_loss={avg_p_loss:.3f}  value_loss={avg_v_loss:.3f}")

        # It's good practice to save checkpoints less frequently, e.g., every 5-10 epochs
        # or based on validation performance, but for now, this is fine.
        checkpoint_save_path = f"models/model4/ckpt_{epoch}.pt"  # Use a consistent path
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_sparse_state_dict': opt_sparse.state_dict(),
            'optimizer_dense_state_dict': opt_dense.state_dict(),
            'avg_p_loss': avg_p_loss,  # Optional: save last loss
            'avg_v_loss': avg_v_loss,  # Optional: save last loss
        }, checkpoint_save_path)
        #torch.save(model.state_dict(), f"weights/ckpt_{epoch}.pt")


if __name__ == "__main__":
    train()