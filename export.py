import torch
import os
import train
from train import VER_NUMBER

# These constants must match the parameters used to train the model you are exporting.
# The GLOBAL_MAX defines the size of the embedding layer's weight matrix.

# Define the output path and ensure the directory exists
output_dir = f"exports/UWTempo/ver{VER_NUMBER}" # Example directory
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Model.onnx")

# --- Model Loading ---
# Instantiate the model with the correct (large) vocabulary size
model = train.Net(train.GLOBAL_MAX, train.ACTIONS_MAX)

# Load the desired checkpoint
checkpoint_path = f"models/model{VER_NUMBER}/ckpt_52.pt" #was 7
try:
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Successfully loaded checkpoint from {checkpoint_path}")
except FileNotFoundError:
    print(f"ERROR: Checkpoint file not found at {checkpoint_path}. Exporting an uninitialized model.")
except Exception as e:
    print(f"ERROR: Failed to load checkpoint. {e}. Exporting an uninitialized model.")

# Set the model to evaluation mode (important for layers like Dropout, etc.)
model.eval()


# 2. Create dummy inputs that match the new model's forward(indices, offsets) signature
#    This represents a single sample (batch size = 1) with 4 active feature indices.
#    The indices must be LongTensors (int64).
dummy_indices = torch.LongTensor([10, 20, 150, 4000]) # Example active feature indices
dummy_offsets = torch.LongTensor([0])                # For a batch of 1, the offset starts at index 0

print("\n--- Running pre-export inference test in PyTorch ---")
with torch.no_grad(): # Disable gradient calculation for inference
    policy_out, value_out = model(dummy_indices, dummy_offsets)
    print("Policy logits (PyTorch):", policy_out)
    print("Value (PyTorch):", value_out)
print("---------------------------------------------------\n")

# The input to the export function must be a tuple containing all positional arguments
dummy_input_tuple = (dummy_indices, dummy_offsets)

# 3. Export to ONNX with updated input names and dynamic axes
print(f"Exporting model to {output_path}...")
torch.onnx.export(
    model,
    dummy_input_tuple,
    output_path,
    input_names=["indices", "offsets"],  # MUST match the names in NeuralNetEvaluator.java
    output_names=["policy", "value"],    # MUST match the names in NeuralNetEvaluator.java
    dynamic_axes={
        "indices": {0: "num_total_indices"}, # The length of this tensor is variable
        "offsets": {0: "batch_size"},        # The batch size is variable
        "policy":  {0: "batch_size"},        # The batch size is variable
        "value":   {0: "batch_size"}         # The batch size is variable
    },
    opset_version=13, # Using a reasonably modern opset is good practice
    do_constant_folding=True
)

print("Model exported successfully.")
print("\n--- ONNX Model Input/Output Summary ---")
print(f"Input 1 Name: 'indices' (dynamic length)")
print(f"Input 2 Name: 'offsets' (dynamic length = batch_size)")
print(f"Output 1 Name: 'policy'")
print(f"Output 2 Name: 'value'")
print("\nEnsure these names match the final String fields in your NeuralNetEvaluator.java")

