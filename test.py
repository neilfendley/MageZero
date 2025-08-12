import torch
import torch.nn.functional as F
from torch import nn  # optim is not strictly needed for testing if not optimizing
from torch.utils.data import DataLoader

# If Net is also in dataset.py or a separate model.py, adjust import accordingly.
import train  # Or from train import Net, ACTIONS_MAX (if ACTIONS_MAX is defined there)
from dataset import LabeledStateDataset, collate_batch  # CRITICAL: Import collate_batch
from train import EPOCH_COUNT


def validate():

    test_ds = "data/UWTempo/ver1/testing/testing1.bin"
    ds = LabeledStateDataset(test_ds)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_batch)

    # 3. Model instantiation uses global vocab size (ds.S) and action size
    model = train.Net(train.GLOBAL_MAX, train.ACTIONS_MAX).cuda()  # should be GLOBAL_VOCAB_SIZE

    model.eval()  # Set model to evaluation mode

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()


    for i in range(1, train.EPOCH_COUNT+1):
        checkpoint_path = f"models/model4/ckpt_{i}.pt"  # Make sure this is the correct checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        except FileNotFoundError:
            print(f"ERROR: Checkpoint file not found at {checkpoint_path}. Testing with an uninitialized model.")
            continue
        except Exception as e:
            print(f"ERROR: Could not load checkpoint. {e}. Testing with an uninitialized model.")
            continue
        # Metrics accumulators
        correct_policy_preds, total_policy_samples = 0, 0
        total_policy_loss, total_value_loss = 0.0, 0.0
        confusion_matrix = torch.zeros(train.ACTIONS_MAX, train.ACTIONS_MAX, dtype=torch.long)

        with torch.no_grad():  # Essential for testing to disable gradient calculations
            # 4. Update the loop to unpack all parts returned by collate_batch
            for batch_indices, batch_offsets, batch_policy_labels, batch_value_labels in dl:
                # 5. Move new input tensors to CUDA
                batch_indices = batch_indices.cuda()
                batch_offsets = batch_offsets.cuda()
                batch_policy_labels = batch_policy_labels.cuda()
                batch_value_labels = batch_value_labels.cuda()

                #Model call uses indices and offsets
                policy_logits, value_pred = model(batch_indices, batch_offsets)
                #Always calc this
                policy_target_indices = torch.argmax(batch_policy_labels, dim=1)

                # Assuming batch_policy_labels are MCTS visit counts/probabilities (distributions).

                lp = ce(policy_logits, batch_policy_labels)


                # Ensure batch_value_labels has the same shape as value_pred ([batch_size])
                lv = mse(value_pred, batch_value_labels.squeeze(-1))
                total_policy_loss += lp.item() * batch_policy_labels.size(0)  # Weighted by actual batch size if last batch is smaller
                total_value_loss += lv.item() * batch_policy_labels.size(0)  # Weighted by actual batch size

                # Accuracy calculation
                predicted_actions = torch.argmax(policy_logits, dim=1)
                # policy_target_indices are already the true action indices
                correct_policy_preds += (predicted_actions == policy_target_indices).sum().item()
                total_policy_samples += batch_policy_labels.size(0)  # Count actual samples in the batch
                # NEW: Populate the confusion matrix
                for true_action, pred_action in zip(policy_target_indices.cpu(), predicted_actions.cpu()):
                    confusion_matrix[true_action, pred_action] += 1

        avg_policy_loss = total_policy_loss / total_policy_samples
        avg_value_loss = total_value_loss / total_policy_samples  # Denominator should be total_policy_samples or len(ds)

        print(f"Test policy_loss={avg_policy_loss:.3f}  value_loss={avg_value_loss:.3f}")
        if total_policy_samples > 0:
            print(f"Test policy_accuracy={correct_policy_preds / total_policy_samples:.3f}")
            # NEW: Print the actual confusion matrix for the first 53 actions
            print("--- Policy Confusion Matrix (True \\ Predicted) ---")
            matrix_size = 53

            # Ensure we don't try to print more than available actions
            if train.ACTIONS_MAX < matrix_size:
                print(
                    f"Warning: ACTIONS_MAX ({train.ACTIONS_MAX}) is smaller than requested matrix size ({matrix_size}). Clamping to ACTIONS_MAX.")
                matrix_size = train.ACTIONS_MAX

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
    validate()