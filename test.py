import torch
import torch.nn.functional as F
from pyroaring import BitMap
from torch import nn  # optim is not strictly needed for testing if not optimizing
from torch.utils.data import DataLoader

# If Net is also in dataset.py or a separate model.py, adjust import accordingly.
import train  # Or from train import Net, ACTIONS_MAX (if ACTIONS_MAX is defined there)
from dataset import AvroIndexed, collate_batch, load_dataset_from_directory, \
    create_redundancy_ignore_list, remove_one_hot_labels  # CRITICAL: Import collate_batch
#from train import Net, ACTIONS_MAX, EPOCH_COUNT, VER_NUMBER, DECK_NAME

SHOW_CONFUSION_MATRIX = True

mse = nn.MSELoss()
kld = nn.KLDivLoss(reduction='batchmean')

def validate(model, dl):
    # Metrics accumulators
    model.eval()
    correct_policy_preds, total_policy_samples = 0, 0
    total_policy_loss, total_value_loss, total_combined_loss = 0.0, 0.0, 0.0
    confusion_matrix = torch.zeros(train.ACTIONS_MAX, train.ACTIONS_MAX, dtype=torch.long)

    with torch.no_grad():  # Essential for testing to disable gradient calculations
        # 4. Update the loop to unpack all parts returned by collate_batch
        for batch_indices, batch_offsets, batch_policy_labels, batch_value_labels in dl:
            # 5. Move new input tensors to CUDA
            batch_indices = batch_indices.cuda()
            batch_offsets = batch_offsets.cuda()
            batch_policy_labels = batch_policy_labels.cuda()
            batch_value_labels = batch_value_labels.cuda()
            # normalize
            batch_policy_labels = train.normalize_policy_labels(batch_policy_labels)

            #Model call uses indices and offsets
            policy_logits, value_pred = model(batch_indices, batch_offsets)

            nonzero = (batch_policy_labels > 0.0001).sum(dim=1)  # [B]
            decision_mask = nonzero > 1  # [B] bool
            #policy loss
            if decision_mask.any():
                logits_d = policy_logits[decision_mask]  # [B_dec, A]
                targets_d = batch_policy_labels[decision_mask]  # [B_dec, A]
                policy_target_indices_d = torch.argmax(batch_policy_labels[decision_mask], dim=1)

                log_probs_d = F.log_softmax(logits_d, dim=1)
                lp = kld(log_probs_d, targets_d)
                b_dec = logits_d.size(0)
                total_policy_loss += lp.item() * b_dec
                total_policy_samples += b_dec

                predicted_actions_d = torch.argmax(policy_logits[decision_mask], dim=1)
                # policy_target_indices are already the true action indices
                correct_policy_preds += (predicted_actions_d == policy_target_indices_d).sum().item()
                # Populate the confusion matrix
                for true_action, pred_action in zip(policy_target_indices_d.cpu(), predicted_actions_d.cpu()):
                    confusion_matrix[true_action, pred_action] += 1
            else:
                lp = torch.zeros((), device=value_pred.device)
            #value loss
            lv = mse(value_pred, batch_value_labels.squeeze(-1))
            #metrics
            total_value_loss += lv.item() * batch_value_labels.size(0)  # Weighted by actual batch size
            combined_loss = lp + lv
            total_combined_loss += combined_loss.item()


    avg_policy_loss = total_policy_loss / max(1,total_policy_samples)
    avg_value_loss = total_value_loss / len(dl.dataset)  # Denominator should be total_policy_samples or len(ds)
    avg_combined_loss = total_combined_loss / len(dl)
    print(f"Test policy_loss={avg_policy_loss:.3f}  value_loss={avg_value_loss:.3f}  decision_states={total_policy_samples}  combined_loss={avg_combined_loss:.3f}")
    if total_policy_samples > 0:
        print(f"Test policy_accuracy={correct_policy_preds / total_policy_samples:.3f}")
        # Print the actual confusion matrix for the first 32 actions
        if SHOW_CONFUSION_MATRIX:
            print("--- Policy Confusion Matrix (True \\ Predicted) ---")
            matrix_size = 32

            sub_matrix = confusion_matrix[:matrix_size, :matrix_size]

            # Print header for predicted actions
            header = "True |" + "".join([f"{j: >4}" for j in range(matrix_size)])
            print(header)
            print("-" * len(header))

            # Print each row for true actions
            for r in range(matrix_size):
                row_str = f"{r: >4} |"
                row_str += "".join([f"{sub_matrix[r, c].item(): >4}" for c in range(matrix_size)])
                print(row_str)

            print("-" * 60)  # Separator for the next checkpoint
        else:
            print("No samples in test set to calculate accuracy.")

if __name__ == "__main__":
    combined_ds = load_dataset_from_directory(f"data/{train.DECK_NAME}/ver{train.VER_NUMBER}/testing")
    with open(f"models/{train.DECK_NAME}/ver{train.VER_NUMBER}/ignore.roar", "rb") as f:
        ignore = BitMap.deserialize(f.read())
    print(len(ignore))
    for ds in combined_ds.datasets:
        ds.ignore_list = ignore

    dl = DataLoader(combined_ds, batch_size=128, shuffle=False, num_workers=16, collate_fn=collate_batch,
                    pin_memory=True, persistent_workers=True)

    # 3. Model instantiation uses global vocab size (ds.S) and action size
    model = train.Net(train.GLOBAL_MAX, train.ACTIONS_MAX).cuda()  # should be GLOBAL_VOCAB_SIZE

    model.eval()  # Set model to evaluation mode


    checkpoint_path = f"models/{train.DECK_NAME}/ver{train.VER_NUMBER}/model.pt"  # Make sure this is the correct checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}. Testing with an uninitialized model.")
    except Exception as e:
        print(f"ERROR: Could not load checkpoint. {e}. Testing with an uninitialized model.")
    validate(model, dl)