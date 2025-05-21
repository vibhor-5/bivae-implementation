import torch
from model import BiVAE
from training_testing import train_bivae, test_model_vectorized
from data_preprocessing import load_interaction_tensor, split_user_interactions
from training_testing import test_model_vectorized

# ----- Config -----
config = {
    "user_encoder_structure": [512],
    "item_encoder_structure": [512],
    "latent_dim": 50,
    "likelihood": "pois",
    "batch_size": 128,
    "epochs": 500,
    "kl_beta": 0.01,
    "learning_rate": 1e-4,
    "log_with": "wandb",
    "device": torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "act_fn": "relu",
    "cap_priors": {"user": False, "item": False},
}

# ----- Load data -----
data_dir = "ml-1m/ratings.dat"
ratings_tensor, user_id_map, item_id_map = load_interaction_tensor(data_dir, binary=True)
train_matrix, val_matrix, test_matrix = split_user_interactions(ratings_tensor, val_ratio=0.1, test_ratio=0.2, seed=42)
num_users, num_items = ratings_tensor.shape
user_inputs = torch.eye(num_users)
item_inputs = torch.eye(num_items)
# ----- Load model -----
user_dim, item_dim = ratings_tensor.shape

def get_ground_truth_dict(test_matrix):
    """
    Converts test interaction matrix to a ground truth dictionary.
    
    Args:
        test_matrix (torch.Tensor): [n_users, n_items] interaction matrix.

    Returns:
        dict: {user_idx: [item indices]}
    """
    test_matrix = test_matrix.to_dense() if test_matrix.is_sparse else test_matrix
    ground_truth_dict = {}

    for user_idx in range(test_matrix.size(0)):
        item_indices = torch.nonzero(test_matrix[user_idx]).squeeze().tolist()
        if isinstance(item_indices, int):  # handle single interaction
            item_indices = [item_indices]
        ground_truth_dict[user_idx] = item_indices

    return ground_truth_dict

model=BiVAE(
    k=config["latent_dim"],
    user_encoder_structure=[num_items] + config["user_encoder_structure"],
    item_encoder_structure=[num_users] + config["item_encoder_structure"],
    act_fn=config["act_fn"],
    likelihood=config["likelihood"],
    cap_priors=config["cap_priors"],
    feature_dim={"user": 0, "item": 0},
    batch_size=config["batch_size"],
).to(config["device"])
model.load_state_dict(torch.load("checkpoints/bivae_final.pt", map_location=config["device"]))
model.eval()

# ----- Run inference -----
print(test_model_vectorized(model, test_matrix, test_matrix.T,get_ground_truth_dict(test_matrix) , device=config["device"]))
