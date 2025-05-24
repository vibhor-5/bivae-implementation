import torch
import wandb
from training_testing import train_bivae,test_model_vectorized
from model import BiVAE
from data_preprocessing import load_interaction_tensor,split_user_interactions
from utils import get_ground_truth_dict 

config = {
    "user_encoder_structure": [512, 256],
    "item_encoder_structure": [512, 256],
    "latent_dim": 100,
    "likelihood": "pois",
    "batch_size": 256,
    "epochs": 200,
    "kl_beta": 0.9,
    "learning_rate": 1e-3,
    "log_with": "wandb",
    "device": torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "act_fn": "elu",
    "cap_priors": {"user": False, "item": False},
    "dropout_rate": 0.2,
    "use_batch_norm": True,
}

data_dir="ml-1m/ratings.dat"
ratings_tensor, user_id_map, item_id_map = load_interaction_tensor(data_dir, binary=False)
train_matrix, val_matrix, test_matrix = split_user_interactions(ratings_tensor, val_ratio=0.1, test_ratio=0.2, seed=42)

model = train_bivae(X_user=train_matrix, val_X_user=val_matrix, **config)
test_model_vectorized(model, test_matrix, test_matrix.T, get_ground_truth_dict(test_matrix), k=10, device=config["device"])


        


