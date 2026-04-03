from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import math
import gzip


import magezero.test as test
from magezero.dataset import H5Indexed, collate_batch,  create_redundancy_ignore_list, filter_opponent_states
from pyroaring import BitMap

def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


#add training data under: data/{deck name}/ver{your version num}/training/{your data}.hdf5
DECK_NAME = env_str("MAGEZERO_DECK_NAME", "IzzetElementals")
VER_NUMBER = env_int("MAGEZERO_VER_NUMBER", 0)

MAKE_IGNORE_LIST = env_bool("MAGEZERO_MAKE_IGNORE_LIST", True)
TRAIN_OPPONENT_HEAD = env_bool("MAGEZERO_TRAIN_OPPONENT_HEAD", False) #turn off when training on round-robin data
ACTIONS_MAX = env_int("MAGEZERO_ACTIONS_MAX", 128)
GLOBAL_MAX = env_int("MAGEZERO_GLOBAL_MAX", 2000000)
EPOCH_COUNT = env_int("MAGEZERO_EPOCH_COUNT", 60)
USE_PREVIOUS_MODEL = env_bool("MAGEZERO_USE_PREVIOUS_MODEL", False)
BATCH_SIZE = env_int("MAGEZERO_BATCH_SIZE", 128)


#TODO: wire into xmage data pipeline
#for now just manually enter your matchup-specific action space sizes here for optimal normalization(XMage prints them at the start of each run)
PRIORITY_A_MAX = env_int("MAGEZERO_PRIORITY_A_MAX", 128)
PRIORITY_B_MAX = env_int("MAGEZERO_PRIORITY_B_MAX", 128)
TARGETS_MAX = env_int("MAGEZERO_TARGETS_MAX", 128)
BINARY_MAX = env_int("MAGEZERO_BINARY_MAX", 2)

def head_weight(K: int) -> float:
    """
    Analytic loss weight to equalize baseline CE scales:
    lambda_K = ln(2) / ln(K)
    """
    if K <= 1:
        raise ValueError("K must be >= 2 for cross-entropy.")
    return math.log(2.0) / math.log(float(K))

#per head weights
lambda_pA = head_weight(PRIORITY_A_MAX)
lambda_pB = head_weight(PRIORITY_B_MAX)
lambda_t = head_weight(TARGETS_MAX)
lambda_b = head_weight(BINARY_MAX)

def load_model(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            out = torch.load(f, map_location=torch.device('cpu'))
            return out

    return torch.load(path)


class ActionType(Enum):
    PRIORITY = 0
    CHOOSE_TARGET = 3
    CHOOSE_USE = 5

"""
MageZero Neural Network architecture for AlphaZero style MCTS:
2M sparse embedding bag -> 512D embedding layer -> 256D hidden layer -> (3 x 128D policy heads + 2D binary policy head + 1D value head)

Policy heads are for each decision type (disjoint action spaces) they are:
128D PriorityA (priority actions for PlayerA - which is this agent)
128D PriorityB (priority actions for PlayerB - which is the opponent)
128D Choose Target (target choices for both players)
2D Choose Use (binary decisions for either player - this is used for selecting attackers and blockers)
"""
class Net(nn.Module):
    def __init__(self, num_embeddings, policy_size_A):
        super().__init__()


        embedding_dim = 512  # Output of EmbeddingBag
        hidden_dim_mlp = 256  # Output of the main MLP block
        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode='sum',
            sparse=True
            #,max_norm=1
        )
        self.embedding_bias = nn.Parameter(torch.zeros(embedding_dim))
        self.input_dropout = 0
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dropout = nn.Dropout(p=0.5)
        self.l1_penalty = None

        self.fc_after_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp),  # From 512 to 256
            nn.ReLU(),
        )
        #policy heads (4 x 256->128 + 1 x 256->2)
        self.player_priority_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.opponent_priority_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.target_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.binary_head = nn.Linear(hidden_dim_mlp, 2)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim_mlp, 1),  # From 256 to 1
            nn.Tanh()
        )

    def forward(self, indices, offsets):
        input_weights = None
        if self.training and self.input_dropout > 0:
            keep_mask = torch.rand_like(indices, dtype=torch.float32) > self.input_dropout
            keep_mask = keep_mask.to(torch.float32)
            input_weights = keep_mask / (1.0 - self.input_dropout)

        emb = self.embedding_bag(indices, offsets, per_sample_weights=input_weights)

        if self.training:
            self.l1_penalty = emb.abs().sum() * 1e-7

        #emb = emb + self.embedding_bias
        #emb = F.relu(emb)
        emb = self.embedding_norm(emb)


        emb = self.embedding_dropout(emb)
        h = self.fc_after_embedding(emb)
        return self.player_priority_head(h), self.opponent_priority_head(h), self.target_head(h), self.binary_head(h), self.value_head(h).squeeze(-1)


def normalize_policy_labels(raw: torch.Tensor) -> torch.Tensor:
    total = raw.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return raw / total


def train():
    os.makedirs(f"models/{DECK_NAME}/ver{VER_NUMBER}", exist_ok=True)
    ds_raw = H5Indexed(f"data/{DECK_NAME}/ver{VER_NUMBER}/training")
    print(torch.cuda.is_available())

    ##Ignore handling
    ignore_list_path = f"models/{DECK_NAME}/ver{VER_NUMBER}/ignore.roar"
    if os.path.exists(ignore_list_path):
        print("Loading existing ignore list from ignore.roar")
        with open(ignore_list_path, "rb") as f:
            loaded_bitmap = BitMap.deserialize(f.read())
        ignore_list = list(loaded_bitmap)
    else:
        print("Generating ignore list from dataset to use for model")
        ignore_list = create_redundancy_ignore_list(ds_raw)
        if not MAKE_IGNORE_LIST: ignore_list = []
        print("Saving ignore list to ignore.roar")

        ignore = BitMap(ignore_list)  # iterable of ints
        with open(f"models/{DECK_NAME}/ver{VER_NUMBER}/ignore.roar", "wb") as f:
            f.write(ignore.serialize())

    # model and data loaders
    model = Net(GLOBAL_MAX, ACTIONS_MAX).cuda()

    # optional start point
    if USE_PREVIOUS_MODEL:
        checkpoint_path = f"models/{DECK_NAME}/ver{VER_NUMBER}/model.pt.gz"
        try:
            #checkpoint = torch.load(checkpoint_path, map_location="cuda")
            checkpoint = load_model(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            with open(f"models/{DECK_NAME}/ver{VER_NUMBER}/ignore.roar", "rb") as f:
                ignore_list2 = BitMap.deserialize(f.read())
                ignore_list.intersection_update(ignore_list2)
                #ignore_list = ignore_list2
            print(f"intersected with previous ignore list: {len(ignore_list2)} for final ignore list: {len(ignore_list)} leaving {GLOBAL_MAX-len(ignore_list)} features")
            # opt_sparse.load_state_dict(checkpoint['optimizer_sparse_state_dict'])
            # opt_dense.load_state_dict(checkpoint['optimizer_dense_state_dict'])
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except FileNotFoundError:
            print(f"INFO: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
        except Exception as e:
            print(f"ERROR: Could not load checkpoint. {e}. Starting from scratch.")

   

    #data sets with redundant filter
    ds = H5Indexed(f"data/{DECK_NAME}/ver{VER_NUMBER}/training", ignore_list)
    test_ds = H5Indexed(f"data/{DECK_NAME}/ver{VER_NUMBER}/testing", ignore_list)

    #if round-robin filter out opponent states AFTER making the ignore list
    if not TRAIN_OPPONENT_HEAD:
        ds = filter_opponent_states(ds,TARGETS_MAX)
        test_ds = filter_opponent_states(test_ds,TARGETS_MAX)



    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_batch,
                    pin_memory=True, persistent_workers=False)

    dl_test = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_batch,
                    pin_memory=True, persistent_workers=False)

    test.SHOW_CONFUSION_MATRIX = False
    breakpoint()
    #optimizers
    opt_sparse = optim.SparseAdam(model.embedding_bag.parameters(), lr=5e-4)
    #opt_sparse = torch.optim.Adagrad(model.embedding_bag.parameters(), lr=0.1,initial_accumulator_value=0.1)
    dense_params = []
    dense_weight_params, dense_bias_params = [], []
    for name, p in model.named_parameters():
        if "embedding_bag" in name:
            continue
        dense_params.append(p)
        if p.ndim > 1:
            dense_weight_params.append(p)
        else:
            dense_bias_params.append(p)

    opt_dense = optim.Adam(dense_params, lr=5e-4)



    mse = nn.MSELoss()
    kld = nn.KLDivLoss(reduction='batchmean')

    #main training loop
    for epoch in range(1, EPOCH_COUNT+1):
        total_pA_loss, total_pB_loss, total_t_loss, total_b_loss, total_v_loss, total_l1_sparse_loss, total_l1_dense_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        total_decision_examples, total_pA_examples, total_pB_examples, total_t_examples, total_b_examples = 0,0,0,0,0
        model.train()

        for batch_indices, batch_offsets, batch_policy_labels, batch_value_labels, is_players, action_types in dl:
            # Move new input tensors to CUDA
            batch_indices = batch_indices.cuda()
            batch_offsets = batch_offsets.cuda()
            batch_policy_labels = batch_policy_labels.cuda()
            batch_value_labels = batch_value_labels.cuda()
            is_players = is_players.cuda().squeeze(-1).to(torch.bool)
            action_types = action_types.cuda().squeeze(-1).to(torch.long)

            # Model call uses indices and offsets
            priority_logits, opponent_priority_logits, target_logits, binary_logits ,value_pred = model(batch_indices, batch_offsets)



            nonzero = (batch_policy_labels > 0).sum(dim=1)  # [B]
            decision_mask = nonzero > 1  # [B] states where more than one action is available
            priority_mask = (action_types==ActionType.PRIORITY.value) & is_players & decision_mask
            opponent_priority_mask = (action_types==ActionType.PRIORITY.value) & (~is_players) & decision_mask
            target_mask = (action_types==ActionType.CHOOSE_TARGET.value) & decision_mask
            binary_mask = (action_types==ActionType.CHOOSE_USE.value) & decision_mask


            total_decision_examples += decision_mask.sum().item()

            #priority A
            if priority_mask.any():
                log_probs_d = F.log_softmax(priority_logits[priority_mask][:,:PRIORITY_A_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[priority_mask][:,:PRIORITY_A_MAX])
                lpA = kld(log_probs_d, tgt)*lambda_pA
                s = log_probs_d.size(0)
                total_pA_loss += lpA.item() * s
                total_pA_examples += s
            else:
                lpA = torch.zeros((), device=value_pred.device)

            #priority B (this uses virtual visits)
            if opponent_priority_mask.any():
                log_probs_d = F.log_softmax(opponent_priority_logits[opponent_priority_mask][:,:PRIORITY_B_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[opponent_priority_mask][:,:PRIORITY_B_MAX])
                lpB = kld(log_probs_d, tgt)*lambda_pB
                s = log_probs_d.size(0)
                total_pB_loss += lpB.item() * s
                total_pB_examples += s
            else:
                lpB = torch.zeros((), device=value_pred.device)

            #targets (shared between both players)
            if target_mask.any():
                log_probs_d = F.log_softmax(target_logits[target_mask][:,:TARGETS_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[target_mask][:,:TARGETS_MAX])
                lt = kld(log_probs_d, tgt)*lambda_t
                s = log_probs_d.size(0)
                total_t_loss += lt.item() * s
                total_t_examples += s
            else:
                lt = torch.zeros((), device=value_pred.device)

            # binary (choose to use) decisions
            if binary_mask.any():
                log_probs_d = F.log_softmax(binary_logits[binary_mask][:,:BINARY_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[binary_mask][:,:BINARY_MAX])
                lb = kld(log_probs_d, tgt)*lambda_b
                s = log_probs_d.size(0)
                total_b_loss += lb.item() * s
                total_b_examples += s
            else:
                lb = torch.zeros((), device=value_pred.device)

            lv = mse(value_pred, batch_value_labels.squeeze(-1))
            l1_dense = 0
            for param in dense_weight_params:
                l1_dense += torch.sum(torch.abs(param))
            l1_dense *= 1e-5

            total_l1_dense_loss += l1_dense.item()
            total_l1_sparse_loss += model.l1_penalty.item()
            loss = lpA + lpB + lt + lb + lv #+ model.l1_penalty #+ l1_dense
            opt_sparse.zero_grad()
            opt_dense.zero_grad()
            loss.backward()
            opt_sparse.step()
            opt_dense.step()


            total_v_loss += lv.item()

        avg_pA_loss = (total_pA_loss / max(total_pA_examples, 1))
        avg_pB_loss = (total_pB_loss / max(total_pB_examples, 1))
        avg_t_loss = (total_t_loss / max(total_t_examples, 1))
        avg_b_loss = (total_b_loss / max(total_b_examples, 1))
        avg_v_loss = total_v_loss / len(dl)
        avg_l1_dense_loss = total_l1_dense_loss / len(dl)
        avg_l1_sparse_loss = total_l1_sparse_loss / len(dl)
        print(f"Epoch {epoch}  priority_A_loss={avg_pA_loss:.3f}  priority_B_loss={avg_pB_loss:.3f} choose_target_loss={avg_t_loss:.3f} choose_use_loss={avg_b_loss:.3f} value_loss={avg_v_loss:.3f} "
              f"l1_dense={avg_l1_dense_loss} l1_sparse={avg_l1_sparse_loss} decision_states={total_decision_examples}")
        #run current model on testing set (if there is one)
        if len(test_ds)>0:
            test.validate(model, dl_test)
        #TODO: make validation based checkpoint schedule
        if epoch == EPOCH_COUNT:
            checkpoint_save_path = f"models/{DECK_NAME}/ver{VER_NUMBER}/model.pt.gz"
            with gzip.open(checkpoint_save_path, 'wb') as f:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_sparse_state_dict': opt_sparse.state_dict(),
                    'optimizer_dense_state_dict': opt_dense.state_dict(),
                    'avg_p_loss': avg_pA_loss,  # Optional: save last losses
                    'avg_v_loss': avg_v_loss,
                }, f)

if __name__ == "__main__":
    train()
