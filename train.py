from training_testing import train_bivae,test_model_vectorized
from model import BiVAE
from data_preprocessing import load_interaction_tensor

config = {
    "user_encoder_structure": [128, 64],                                                                    #
    "item_encoder_structure": [128, 64],
    "latent_dim": 32,
    "likelihood": "bern",
    "batch_size": 128,
    "epochs": 100,
    "k": 32,
    "beta": 0.1,
    "theta": 0.1,
    "mu_beta": 0.1,
    "mu_theta": 0.1,
    "kl_beta": 0.2,
    "learning_rate": 1e-3,
    "log_with":"tensorboard",
    "device": torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "act_fn": "relu",
    "cap_priors": {"user": False, "item": False},
}

data_dir="ml-1m/ratings.csv"
ratings_tensor, user_id_map, item_id_map = load_interaction_tensor(data_dir, binary=True)
train_matrix, val_matrix, test_matrix = split_user_interactions(ratings_tensor, val_ratio=0.1, test_ratio=0.2, seed=42)

train_bivae(X_user=train_matrix,val_X_user=val_matrix,config**)
test_model_vectorized(model, test_matrix, user_id_map, item_id_map, config["device"])


        


