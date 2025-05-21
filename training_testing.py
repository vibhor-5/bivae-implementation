import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from datetime import datetime
from model import BiVAE

def train_bivae(
    X_user,
    user_features=None,
    item_features=None,
    val_X_user=None,
    val_user_features=None,
    val_item_features=None,
    latent_dim=50,
    user_encoder_structure=[400, 200],
    item_encoder_structure=[400, 200],
    act_fn="tanh",
    likelihood="bern",
    cap_priors={"user": False, "item": False},
    feature_dim={"user": 0, "item": 0},
    batch_size=128,
    epochs=50,
    kl_beta=0.2,
    learning_rate=1e-3,
    device=None,
    verbose=True,
    log_with=None,  # 'tensorboard', 'wandb', or None
    log_dir="logs/",
    checkpoint_dir="checkpoints/",
    project_name="BiVAE"
):
    assert isinstance(X_user, torch.Tensor), "X_user must be a torch.Tensor"
    X_item = X_user.T
    n_users, n_items = X_user.shape

    if val_X_user is not None:
        assert isinstance(val_X_user, torch.Tensor), "val_X_user must be a torch.Tensor"
        val_X_item = val_X_user.T

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_encoder_structure = [n_items] + user_encoder_structure
    item_encoder_structure = [n_users] + item_encoder_structure

    model = BiVAE(
        k=latent_dim,
        user_encoder_structure=user_encoder_structure,
        item_encoder_structure=item_encoder_structure,
        act_fn=act_fn,
        likelihood=likelihood,
        cap_priors=cap_priors,
        feature_dim=feature_dim,
        batch_size=batch_size,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Logging setup
    writer = None
    if log_with == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        log_path = os.path.join(log_dir, f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        writer = SummaryWriter(log_path)
    elif log_with == "wandb":
        import wandb
        wandb.init(project=project_name, config={
            "latent_dim": latent_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "kl_beta": kl_beta,
            "cap_priors": cap_priors,
            "likelihood": likelihood,
        })
        wandb.watch(model)

    user_loader = DataLoader(TensorDataset(X_user), batch_size=batch_size, shuffle=True)
    item_loader = DataLoader(TensorDataset(X_item), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_user_loss = total_item_loss = 0.0

        for batch in user_loader:
            x_user = batch[0].to(device)
            theta, x_recon, mu, std = model(x_user, user=True, beta=model.beta)

            if cap_priors.get("user", False) and user_features is not None:
                mu_prior = model.encode_user_prior(user_features[:x_user.size(0)].to(device))
            else:
                mu_prior = torch.zeros_like(mu)

            loss = model.loss(x_user, x_recon, mu, mu_prior, std, kl_beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_user_loss += loss.item()

        for batch in item_loader:
            x_item = batch[0].to(device)
            beta, x_recon, mu, std = model(x_item, user=False, theta=model.theta)

            if cap_priors.get("item", False) and item_features is not None:
                mu_prior = model.encode_item_prior(item_features[:x_item.size(0)].to(device))
            else:
                mu_prior = torch.zeros_like(mu)

            loss = model.loss(x_item, x_recon, mu, mu_prior, std, kl_beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_item_loss += loss.item()

        avg_user_loss = total_user_loss / len(user_loader)
        avg_item_loss = total_item_loss / len(item_loader)
        avg_val_loss = 0.0

        # --- Validation ---
        if val_X_user is not None:
            model.eval()
            with torch.no_grad():
                val_user_loader = DataLoader(TensorDataset(val_X_user), batch_size=batch_size)
                val_item_loader = DataLoader(TensorDataset(val_X_item), batch_size=batch_size)

                for batch in val_user_loader:
                    x_val = batch[0].to(device)
                    theta, x_recon, mu, std = model(x_val, user=True, beta=model.beta)
                    if cap_priors.get("user", False) and val_user_features is not None:
                        mu_prior = model.encode_user_prior(val_user_features[:x_val.size(0)].to(device))
                    else:
                        mu_prior = torch.zeros_like(mu)
                    val_loss = model.loss(x_val, x_recon, mu, mu_prior, std, kl_beta)
                    avg_val_loss += val_loss.item()

                for batch in val_item_loader:
                    x_val = batch[0].to(device)
                    beta, x_recon, mu, std = model(x_val, user=False, theta=model.theta)
                    if cap_priors.get("item", False) and val_item_features is not None:
                        mu_prior = model.encode_item_prior(val_item_features[:x_val.size(0)].to(device))
                    else:
                        mu_prior = torch.zeros_like(mu)
                    val_loss = model.loss(x_val, x_recon, mu, mu_prior, std, kl_beta)
                    avg_val_loss += val_loss.item()

                avg_val_loss /= (len(val_user_loader) + len(val_item_loader))

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(checkpoint_dir, f"bivae_best.pt")
                    torch.save(model.state_dict(), best_model_path)

        if log_with == "tensorboard" and writer:
            writer.add_scalar("Loss/Train_User", avg_user_loss, epoch)
            writer.add_scalar("Loss/Train_Item", avg_item_loss, epoch)
            if val_X_user is not None:
                writer.add_scalar("Loss/Val", avg_val_loss, epoch)

        if log_with == "wandb":
            log_dict = {
                "train_user_loss": avg_user_loss,
                "train_item_loss": avg_item_loss,
            }
            if val_X_user is not None:
                log_dict["val_loss"] = avg_val_loss
            wandb.log(log_dict)

        if verbose:
            msg = f"Epoch {epoch+1}/{epochs} | Train Loss (U/I): {avg_user_loss:.4f}/{avg_item_loss:.4f}"
            if val_X_user is not None:
                msg += f" | Val Loss: {avg_val_loss:.4f}"
            print(msg)
        
        if epoch%10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"bivae_epoch_{epoch}.pt"))
    
    # Final save
    final_model_path = os.path.join(checkpoint_dir, f"bivae_final.pt")
    torch.save(model.state_dict(), final_model_path)

    if log_with == "tensorboard" and writer:
        writer.close()
    if log_with == "wandb":
        wandb.finish()

    

    # Load best model if saved
    # best_model_path = os.path.join(checkpoint_dir, f"bivae_best.pt")
    # if os.path.exists(best_model_path):
    #     model.load_state_dict(torch.load(best_model_path))

    return model


def compute_ndcg_at_k(ranked_indices, ground_truth, k):
    """
    Vectorized NDCG@k computation.

    ranked_indices: [n_users, k] array of top-k predicted item indices
    ground_truth: list of sets, each set contains ground truth items for that user
    """
    gains = np.zeros(ranked_indices.shape)
    for i, gt_items in enumerate(ground_truth):
        for j in range(k):
            if ranked_indices[i, j] in gt_items:
                gains[i, j] = 1.0 / np.log2(j + 2)
    dcg = np.sum(gains, axis=1)
    idcg = np.array([sum([1.0 / np.log2(i + 2) for i in range(min(len(gt), k))]) for gt in ground_truth])
    idcg[idcg == 0] = 1e-10  # prevent division by zero
    return dcg / idcg

def compute_hr_at_k(ranked_indices, ground_truth, k):
    """
    Vectorized HR@k computation.
    """
    hits = np.zeros(ranked_indices.shape[0])
    for i, gt_items in enumerate(ground_truth):
        hits[i] = any(item in gt_items for item in ranked_indices[i, :k])
    return hits

def test_model_vectorized(model, user_inputs, item_inputs, ground_truth_dict, k=10, device="cpu"):
    """
    Vectorized BiVAE test with NDCG@k and HR@k.

    Parameters:
    - model: trained BiVAE model
    - user_inputs: Tensor [n_users, user_feat_dim]
    - item_inputs: Tensor [n_items, item_feat_dim]
    - ground_truth_dict: dict {user_idx: [item indices]}
    - k: cutoff
    - device: device for computation

    Returns:
    - avg_ndcg@k, avg_hr@k
    """
    model.eval()
    model.to(device)
    user_inputs = user_inputs.to(device)
    item_inputs = item_inputs.to(device)

    with torch.no_grad():
        user_mu, user_std = model.encode_user(user_inputs)
        item_mu, item_std = model.encode_item(item_inputs)

        theta = model.reparameterize(user_mu, user_std)   # [n_users, k]
        beta = model.reparameterize(item_mu, item_std)    # [n_items, k]

        scores = torch.sigmoid(theta @ beta.T).cpu().numpy()  # [n_users, n_items]
        top_k_indices = np.argsort(-scores, axis=1)[:, :k]    # Top-k indices per user

    # Prepare ground truth as list of sets
    n_users = user_inputs.shape[0]
    ground_truth = [set(ground_truth_dict.get(i, [])) for i in range(n_users)]

    # Compute metrics
    ndcg_scores = compute_ndcg_at_k(top_k_indices, ground_truth, k)
    hr_scores = compute_hr_at_k(top_k_indices, ground_truth, k)

    return np.mean(ndcg_scores), np.mean(hr_scores)
