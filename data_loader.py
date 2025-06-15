import torch
import numpy as np
from scipy import sparse
from typing import Iterator, Tuple, Optional
import os
import pandas as pd
from sklearn.model_selection import train_test_split


class BiVAEDataLoader:
    """
    Data loader for BiVAE model
    Handles data loading, preprocessing, and batching
    """
    
    def __init__(
        self,
        data_path: str,
        min_rating: float = 4.0,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize data loader
        
        Args:
            data_path: Path to MovieLens data
            min_rating: Minimum rating for implicit feedback
            test_size: Test set ratio
            random_state: Random seed
        """
        self.data_path = data_path
        self.min_rating = min_rating
        self.test_size = test_size
        self.random_state = random_state
        
        # Load and preprocess data
        self._load_data()
        self._preprocess_data()
    
    def _load_data(self):
        """Load MovieLens data"""
        # Load ratings
        ratings_file = os.path.join(self.data_path, 'ratings.dat')
        ratings = pd.read_csv(
            ratings_file,
            sep='::',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        # Create user and movie ID mappings
        unique_users = ratings['user_id'].unique()
        unique_movies = ratings['movie_id'].unique()
        
        self.user_id_map = {uid: i for i, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: i for i, mid in enumerate(unique_movies)}
        
        # Convert to 0-based indexing
        ratings['user_id'] = ratings['user_id'].map(self.user_id_map)
        ratings['movie_id'] = ratings['movie_id'].map(self.movie_id_map)
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_movies)
        
        # Create sparse matrix
        self.ratings_matrix = sparse.csr_matrix(
            (ratings['rating'], (ratings['user_id'], ratings['movie_id'])),
            shape=(self.num_users, self.num_items)
        )
    
    def _preprocess_data(self):
        """Preprocess data for training"""
        # Convert to binary implicit feedback
        binary_matrix = self.ratings_matrix.copy()
        binary_matrix.data = (binary_matrix.data >= self.min_rating).astype(np.float32)
        
        # Split into train and test
        train_matrix, test_matrix = self._split_train_test(binary_matrix)
        
        self.train_matrix = train_matrix
        self.test_matrix = test_matrix
    
    def _split_train_test(self, matrix: sparse.csr_matrix) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Split data into train and test sets maintaining user presence in both sets
        
        Args:
            matrix: Input sparse matrix
            
        Returns:
            Tuple of (train_matrix, test_matrix)
        """
        train_matrix = matrix.copy()
        test_matrix = matrix.copy()
        
        # For each user, randomly select items for test set
        for user_idx in range(self.num_users):
            user_items = matrix[user_idx].nonzero()[1]
            if len(user_items) > 1:  # Only split if user has more than one item
                train_items, test_items = train_test_split(
                    user_items,
                    test_size=self.test_size,
                    random_state=self.random_state
                )
                
                # Update matrices
                train_matrix[user_idx, test_items] = 0
                test_matrix[user_idx, train_items] = 0
        
        return train_matrix, test_matrix
    
    def user_iter(self, batch_size: int, shuffle: bool = True) -> Iterator[np.ndarray]:
        """
        Iterate over users in batches
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle users
            
        Yields:
            Batches of user indices
        """
        indices = np.arange(self.num_users)
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.num_users, batch_size):
            yield indices[i:i + batch_size]
    
    def item_iter(self, batch_size: int, shuffle: bool = True) -> Iterator[np.ndarray]:
        """
        Iterate over items in batches
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle items
            
        Yields:
            Batches of item indices
        """
        indices = np.arange(self.num_items)
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.num_items, batch_size):
            yield indices[i:i + batch_size]
    
    def get_user_batch(self, user_ids: np.ndarray) -> torch.Tensor:
        """
        Get batch of user interactions
        
        Args:
            user_ids: Array of user indices
            
        Returns:
            Tensor of user interactions
        """
        batch = self.train_matrix[user_ids].toarray()
        return torch.tensor(batch, dtype=torch.float32)
    
    def get_item_batch(self, item_ids: np.ndarray) -> torch.Tensor:
        """
        Get batch of item interactions
        
        Args:
            item_ids: Array of item indices
            
        Returns:
            Tensor of item interactions
        """
        batch = self.train_matrix.t()[item_ids].toarray()
        return torch.tensor(batch, dtype=torch.float32)
    
    def get_original_movie_id(self, mapped_id: int) -> int:
        """
        Convert mapped movie ID back to original ID
        
        Args:
            mapped_id: Mapped movie ID
            
        Returns:
            Original movie ID
        """
        reverse_map = {v: k for k, v in self.movie_id_map.items()}
        return reverse_map[mapped_id]
    
    def get_original_user_id(self, mapped_id: int) -> int:
        """
        Convert mapped user ID back to original ID
        
        Args:
            mapped_id: Mapped user ID
            
        Returns:
            Original user ID
        """
        reverse_map = {v: k for k, v in self.user_id_map.items()}
        return reverse_map[mapped_id] 