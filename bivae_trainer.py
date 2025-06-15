import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import itertools as it
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import os
from datetime import datetime

class BiVAETrainer:
    """
    Trainer class for BiVAE model with comprehensive training and evaluation utilities
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_loader: object,
        learning_rate: float = 0.001,
        kl_beta: float = 1.0,
        batch_size: int = 500,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints/",
        project_name: str = "BiVAE"
    ):
        """
        Initialize BiVAE trainer
        
        Args:
            model: BiVAE model instance
            data_loader: DataLoader instance for training data
            learning_rate: Learning rate for optimizers
            kl_beta: Weight for KL divergence term
            batch_size: Batch size for training
            device: Device to use for training (auto-detected if None)
            checkpoint_dir: Directory to save model checkpoints
            project_name: Name of the project for logging
        """
        self.model = model
        self.data_loader = data_loader
        self.learning_rate = learning_rate
        self.kl_beta = kl_beta
        self.batch_size = batch_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.project_name = project_name
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training history
        self.history = {
            'user_loss': [],
            'item_loss': [],
            'total_loss': [],
            'precision_at_k': [],
            'recall_at_k': [],
            'f1_at_k': []
        }
        
        # Setup optimizers
        self._setup_optimizers()
        
        print(f"Training on device: {self.device}")
    
    def _setup_optimizers(self):
        """Setup separate optimizers for user and item parameters"""
        # User parameters
        user_params = it.chain(
            self.model.user_encoder.parameters(),
            self.model.user_mu.parameters(),
            self.model.user_std.parameters(),
        )
        
        # Item parameters
        item_params = it.chain(
            self.model.item_encoder.parameters(),
            self.model.item_mu.parameters(),
            self.model.item_std.parameters(),
        )
        
        # Add prior encoders if using content-aware priors
        if self.model.cap_priors.get("user", False):
            user_params = it.chain(user_params, self.model.user_prior_encoder.parameters())
        
        if self.model.cap_priors.get("item", False):
            item_params = it.chain(item_params, self.model.item_prior_encoder.parameters())
        
        self.user_optimizer = optim.Adam(user_params, lr=self.learning_rate)
        self.item_optimizer = optim.Adam(item_params, lr=self.learning_rate)
        
        # Learning rate schedulers
        self.user_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.user_optimizer, mode='min', patience=10, factor=0.5
        )
        self.item_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.item_optimizer, mode='min', patience=10, factor=0.5
        ) 

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train one epoch alternating between user and item sides
        
        Returns:
            Tuple of (average user loss, average item loss)
        """
        self.model.train()
        
        # Get interaction matrices
        train_matrix = self.data_loader.train_matrix
        train_matrix_t = train_matrix.transpose()
        
        # Convert to binary
        train_matrix_binary = train_matrix.copy()
        train_matrix_binary.data = np.ones_like(train_matrix_binary.data)
        train_matrix_t_binary = train_matrix_binary.transpose()
        
        total_user_loss = 0.0
        total_item_loss = 0.0
        user_count = 0
        item_count = 0
        
        # Train item side
        for item_ids in self.data_loader.item_iter(self.batch_size, shuffle=True):
            item_batch = train_matrix_t_binary[item_ids, :].toarray()
            item_batch = torch.tensor(item_batch, dtype=torch.float32, device=self.device)
            
            # Forward pass
            beta, item_recon, item_mu, item_std = self.model(
                item_batch, user=False, theta=self.model.theta
            )
            
            # Prior (zero mean for standard normal prior)
            item_mu_prior = torch.zeros_like(item_mu)
            if self.model.cap_priors.get("item", False):
                # Would use item features here if available
                pass
            
            # Compute loss
            item_loss = self.model.loss(
                item_batch, item_recon, item_mu, item_mu_prior, item_std, self.kl_beta
            )
            
            # Backward pass
            self.item_optimizer.zero_grad()
            item_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.item_encoder.parameters(), max_norm=1.0)
            self.item_optimizer.step()
            
            total_item_loss += item_loss.item()
            item_count += len(item_batch)
            
            # Update item embeddings
            with torch.no_grad():
                beta, _, item_mu, _ = self.model(
                    item_batch, user=False, theta=self.model.theta
                )
                self.model.beta.data[item_ids] = beta.data
                self.model.mu_beta.data[item_ids] = item_mu.data
        
        # Train user side
        for user_ids in self.data_loader.user_iter(self.batch_size, shuffle=True):
            user_batch = train_matrix_binary[user_ids, :].toarray()
            user_batch = torch.tensor(user_batch, dtype=torch.float32, device=self.device)
            
            # Forward pass
            theta, user_recon, user_mu, user_std = self.model(
                user_batch, user=True, beta=self.model.beta
            )
            
            # Prior (zero mean for standard normal prior)
            user_mu_prior = torch.zeros_like(user_mu)
            if self.model.cap_priors.get("user", False):
                # Would use user features here if available
                pass
            
            # Compute loss
            user_loss = self.model.loss(
                user_batch, user_recon, user_mu, user_mu_prior, user_std, self.kl_beta
            )
            
            # Backward pass
            self.user_optimizer.zero_grad()
            user_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.user_encoder.parameters(), max_norm=1.0)
            self.user_optimizer.step()
            
            total_user_loss += user_loss.item()
            user_count += len(user_batch)
            
            # Update user embeddings
            with torch.no_grad():
                theta, _, user_mu, _ = self.model(
                    user_batch, user=True, beta=self.model.beta
                )
                self.model.theta.data[user_ids] = theta.data
                self.model.mu_theta.data[user_ids] = user_mu.data
        
        # Final update of embeddings
        self._update_final_embeddings()
        
        avg_user_loss = total_user_loss / user_count if user_count > 0 else 0
        avg_item_loss = total_item_loss / item_count if item_count > 0 else 0
        
        return avg_user_loss, avg_item_loss
    
    def _update_final_embeddings(self):
        """Update final embeddings after training epoch"""
        train_matrix = self.data_loader.train_matrix
        train_matrix_binary = train_matrix.copy()
        train_matrix_binary.data = np.ones_like(train_matrix_binary.data)
        train_matrix_t_binary = train_matrix_binary.transpose()
        
        self.model.eval()
        with torch.no_grad():
            # Update item embeddings
            for item_ids in self.data_loader.item_iter(self.batch_size, shuffle=False):
                item_batch = train_matrix_t_binary[item_ids, :].toarray()
                item_batch = torch.tensor(item_batch, dtype=torch.float32, device=self.device)
                
                _, _, item_mu, _ = self.model(
                    item_batch, user=False, theta=self.model.theta
                )
                self.model.mu_beta.data[item_ids] = item_mu.data
            
            # Update user embeddings
            for user_ids in self.data_loader.user_iter(self.batch_size, shuffle=False):
                user_batch = train_matrix_binary[user_ids, :].toarray()
                user_batch = torch.tensor(user_batch, dtype=torch.float32, device=self.device)
                
                _, _, user_mu, _ = self.model(
                    user_batch, user=True, beta=self.model.beta
                )
                self.model.mu_theta.data[user_ids] = user_mu.data 

    def evaluate(self, k: int = 10) -> Dict[str, float]:
        """
        Evaluate model performance using precision, recall, and F1 at k
        
        Args:
            k: Number of top items to consider for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        train_matrix = self.data_loader.train_matrix
        test_matrix = self.data_loader.test_matrix
        
        if test_matrix.nnz == 0:
            print("No test data available for evaluation")
            return {
                f'precision_at_{k}': 0.0,
                f'recall_at_{k}': 0.0,
                f'f1_at_{k}': 0.0
            }
        
        precisions = []
        recalls = []
        f1s = []
        
        with torch.no_grad():
            # Get predictions
            predictions = self.model.predict().cpu().numpy()
            
            for user_idx in range(self.data_loader.num_users):
                # Get test items for this user
                test_items = test_matrix[user_idx].nonzero()[1]
                if len(test_items) == 0:
                    continue
                
                # Get train items to exclude from recommendations
                train_items = train_matrix[user_idx].nonzero()[1]
                
                # Get user predictions and exclude training items
                user_preds = predictions[user_idx].copy()
                user_preds[train_items] = -np.inf
                
                # Get top-k recommendations
                top_k_items = np.argsort(user_preds)[-k:]
                
                # Calculate metrics
                relevant_items = set(test_items)
                recommended_items = set(top_k_items)
                
                if len(recommended_items) > 0:
                    precision = len(relevant_items & recommended_items) / len(recommended_items)
                    recall = len(relevant_items & recommended_items) / len(relevant_items)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)
        
        return {
            f'precision_at_{k}': np.mean(precisions),
            f'recall_at_{k}': np.mean(recalls),
            f'f1_at_{k}': np.mean(f1s)
        }
    
    def train(
        self,
        num_epochs: int = 100,
        eval_every: int = 10,
        verbose: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the BiVAE model
        
        Args:
            num_epochs: Number of training epochs
            eval_every: Evaluate every N epochs
            verbose: Whether to print progress
            save_path: Path to save the best model
            
        Returns:
            Dictionary of training history
        """
        best_val_f1 = 0.0
        early_stopping = EarlyStopping(patience=20, min_delta=0.001)
        
        for epoch in range(num_epochs):
            # Train one epoch
            user_loss, item_loss = self.train_epoch()
            total_loss = user_loss + item_loss
            
            # Update history
            self.history['user_loss'].append(user_loss)
            self.history['item_loss'].append(item_loss)
            self.history['total_loss'].append(total_loss)
            
            # Evaluate
            if (epoch + 1) % eval_every == 0:
                metrics = self.evaluate()
                self.history['precision_at_k'].append(metrics['precision_at_10'])
                self.history['recall_at_k'].append(metrics['recall_at_10'])
                self.history['f1_at_k'].append(metrics['f1_at_10'])
                
                # Update learning rates
                self.user_scheduler.step(total_loss)
                self.item_scheduler.step(total_loss)
                
                # Save best model
                if metrics['f1_at_10'] > best_val_f1:
                    best_val_f1 = metrics['f1_at_10']
                    if save_path:
                        self.save_model(save_path)
                
                # Early stopping
                if early_stopping(metrics['f1_at_10'], self.model):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Print progress
            if verbose:
                msg = f"Epoch {epoch + 1}/{num_epochs} | "
                msg += f"User Loss: {user_loss:.4f} | Item Loss: {item_loss:.4f} | "
                msg += f"Total Loss: {total_loss:.4f}"
                if (epoch + 1) % eval_every == 0:
                    msg += f" | F1@10: {metrics['f1_at_10']:.4f}"
                print(msg)
        
        return self.history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.history['user_loss'], label='User Loss')
        plt.plot(self.history['item_loss'], label='Item Loss')
        plt.plot(self.history['total_loss'], label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses')
        
        # Plot metrics
        plt.subplot(1, 2, 2)
        plt.plot(self.history['precision_at_k'], label='Precision@10')
        plt.plot(self.history['recall_at_k'], label='Recall@10')
        plt.plot(self.history['f1_at_k'], label='F1@10')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Evaluation Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_optimizer_state_dict': self.user_optimizer.state_dict(),
            'item_optimizer_state_dict': self.item_optimizer.state_dict(),
            'user_scheduler_state_dict': self.user_scheduler.state_dict(),
            'item_scheduler_state_dict': self.item_scheduler.state_dict(),
            'history': self.history
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model from checkpoint"""
        try:
            # First try loading with weights_only=False for backward compatibility
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            # If that fails, try with weights_only=True and add safe globals
            torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        # Load model and optimizer state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.user_optimizer.load_state_dict(checkpoint['user_optimizer_state_dict'])
        self.item_optimizer.load_state_dict(checkpoint['item_optimizer_state_dict'])
        self.user_scheduler.load_state_dict(checkpoint['user_scheduler_state_dict'])
        self.item_scheduler.load_state_dict(checkpoint['item_scheduler_state_dict'])
        
        # Load optional checkpoint data with defaults
        self.history = checkpoint.get('history', {})
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_ndcg = checkpoint.get('best_ndcg', 0.0)


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop 