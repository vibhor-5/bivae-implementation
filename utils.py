import torch
import numpy as np

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