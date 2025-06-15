import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

EPS = 1e-10

ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}


class BiVAE(nn.Module):
    """
    Bilateral Variational Autoencoder for Collaborative Filtering
    
    This model learns user and item representations jointly using variational autoencoders.
    It implements a bilateral architecture where both users and items are encoded into
    latent representations that are used to reconstruct the interaction matrix.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        k: int = 50,
        user_encoder_structure: Optional[List[int]] = None,
        item_encoder_structure: Optional[List[int]] = None,
        act_fn: str = "tanh",
        likelihood: str = "bern",
        cap_priors: Optional[Dict[str, bool]] = None,
        feature_dim: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize BiVAE model
        
        Args:
            num_users: Number of users
            num_items: Number of items
            k: Latent dimension
            user_encoder_structure: List of layer sizes for user encoder
            item_encoder_structure: List of layer sizes for item encoder
            act_fn: Activation function name
            likelihood: Likelihood function name
            cap_priors: Dictionary indicating whether to use content-aware priors
            feature_dim: Dictionary of feature dimensions for content-aware priors
        """
        super(BiVAE, self).__init__()
        
        # Default encoder structures
        if user_encoder_structure is None:
            user_encoder_structure = [num_items, 600, 200]
        if item_encoder_structure is None:
            item_encoder_structure = [num_users, 600, 200]
        if cap_priors is None:
            cap_priors = {"user": False, "item": False}
        if feature_dim is None:
            feature_dim = {"user": 0, "item": 0}
            
        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        self.likelihood = likelihood
        self.cap_priors = cap_priors
        
        # Initialize latent variables
        self.mu_theta = torch.zeros((num_users, k))  # user means
        self.mu_beta = torch.zeros((num_items, k))   # item means
        self.theta = torch.randn(num_users, k) * 0.01  # user embeddings
        self.beta = torch.randn(num_items, k) * 0.01   # item embeddings
        
        # Initialize user embeddings with Kaiming uniform
        torch.nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.beta, a=np.sqrt(5))
        
        # Activation function
        self.act_fn = ACT.get(act_fn, None)
        if self.act_fn is None:
            raise ValueError(f"Supported act_fn: {list(ACT.keys())}")
        
        # Prior encoders for content-aware priors
        if self.cap_priors.get("user", False):
            self.user_prior_encoder = nn.Linear(feature_dim["user"], k)
        if self.cap_priors.get("item", False):
            self.item_prior_encoder = nn.Linear(feature_dim["item"], k)
        
        # User Encoder
        self.user_encoder = nn.Sequential()
        for i in range(len(user_encoder_structure) - 1):
            self.user_encoder.add_module(
                f"fc{i}",
                nn.Linear(user_encoder_structure[i], user_encoder_structure[i + 1]),
            )
            if i < len(user_encoder_structure) - 2:  # No activation after last layer
                self.user_encoder.add_module(f"act{i}", self.act_fn)
                
        self.user_mu = nn.Linear(user_encoder_structure[-1], k)
        self.user_std = nn.Linear(user_encoder_structure[-1], k)
        
        # Item Encoder
        self.item_encoder = nn.Sequential()
        for i in range(len(item_encoder_structure) - 1):
            self.item_encoder.add_module(
                f"fc{i}",
                nn.Linear(item_encoder_structure[i], item_encoder_structure[i + 1]),
            )
            if i < len(item_encoder_structure) - 2:  # No activation after last layer
                self.item_encoder.add_module(f"act{i}", self.act_fn)
                
        self.item_mu = nn.Linear(item_encoder_structure[-1], k)
        self.item_std = nn.Linear(item_encoder_structure[-1], k)
    
    def to(self, device: torch.device) -> 'BiVAE':
        """
        Move model and parameters to device
        
        Args:
            device: Device to move model to
            
        Returns:
            Model instance
        """
        self.beta = self.beta.to(device=device)
        self.theta = self.theta.to(device=device)
        self.mu_beta = self.mu_beta.to(device=device)
        self.mu_theta = self.mu_theta.to(device=device)
        return super(BiVAE, self).to(device)
    
    def encode_user_prior(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode user content features to prior mean
        
        Args:
            x: User content features
            
        Returns:
            Prior mean for user
        """
        return self.user_prior_encoder(x)
    
    def encode_item_prior(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode item content features to prior mean
        
        Args:
            x: Item content features
            
        Returns:
            Prior mean for item
        """
        return self.item_prior_encoder(x)
    
    def encode_user(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode user interaction history to latent parameters
        
        Args:
            x: User interaction vector
            
        Returns:
            Tuple of (mean, standard deviation)
        """
        h = self.user_encoder(x)
        return self.user_mu(h), torch.sigmoid(self.user_std(h))
    
    def encode_item(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode item interaction history to latent parameters
        
        Args:
            x: Item interaction vector
            
        Returns:
            Tuple of (mean, standard deviation)
        """
        h = self.item_encoder(x)
        return self.item_mu(h), torch.sigmoid(self.item_std(h))
    
    def decode_user(self, theta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Decode user latent variables to user preferences
        
        Args:
            theta: User latent variables
            beta: Item latent variables
            
        Returns:
            Reconstructed user preferences
        """
        h = theta.mm(beta.t())
        return torch.sigmoid(h)
    
    def decode_item(self, theta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Decode item latent variables to item preferences
        
        Args:
            theta: User latent variables
            beta: Item latent variables
            
        Returns:
            Reconstructed item preferences
        """
        h = beta.mm(theta.t())
        return torch.sigmoid(h)
    
    def reparameterize(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from Gaussian
        
        Args:
            mu: Mean
            std: Standard deviation
            
        Returns:
            Sampled latent variables
        """
        eps = torch.randn_like(mu)
        return mu + eps * std
    
    def forward(
        self,
        x: torch.Tensor,
        user: bool = True,
        beta: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            x: Input interaction vector
            user: If True, process as user, else as item
            beta: Item embeddings (for user forward pass)
            theta: User embeddings (for item forward pass)
            
        Returns:
            Tuple of (latent variables, reconstruction, mean, standard deviation)
        """
        if user:
            mu, std = self.encode_user(x)
            theta = self.reparameterize(mu, std)
            return theta, self.decode_user(theta, beta), mu, std
        else:
            mu, std = self.encode_item(x)
            beta = self.reparameterize(mu, std)
            return beta, self.decode_item(theta, beta), mu, std
    
    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        mu_prior: torch.Tensor,
        std: torch.Tensor,
        kl_beta: float
    ) -> torch.Tensor:
        """
        Compute the loss function (negative ELBO)
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Encoded mean
            mu_prior: Prior mean
            std: Encoded standard deviation
            kl_beta: KL divergence weight
            
        Returns:
            Loss value
        """
        # Likelihood term
        ll_choices = {
            "bern": x * torch.log(x_recon + EPS) + (1 - x) * torch.log(1 - x_recon + EPS),
            "gaus": -((x - x_recon) ** 2),
            "pois": x * torch.log(x_recon + EPS) - x_recon,
        }
        
        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError(f"Supported likelihoods: {list(ll_choices.keys())}")
        
        ll = torch.sum(ll, dim=1)
        
        # KL divergence term
        kld = -0.5 * (1 + 2.0 * torch.log(std) - (mu - mu_prior).pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)
        
        return torch.mean(kl_beta * kld - ll)
    
    def predict(
        self,
        user_ids: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate predictions for user-item pairs
        
        Args:
            user_ids: List of user IDs (if None, predict for all users)
            item_ids: List of item IDs (if None, predict for all items)
            
        Returns:
            Prediction matrix
        """
        if user_ids is None:
            user_embeddings = self.mu_theta
        else:
            user_embeddings = self.mu_theta[user_ids]
            
        if item_ids is None:
            item_embeddings = self.mu_beta
        else:
            item_embeddings = self.mu_beta[item_ids]
            
        return torch.sigmoid(user_embeddings @ item_embeddings.t())