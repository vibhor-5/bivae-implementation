import argparse
import torch
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple
import os
from data_loader import BiVAEDataLoader
from bivae_model import BiVAE
from bivae_trainer import BiVAETrainer


def calculate_ndcg(predictions: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """
    Calculate NDCG@k
    
    Args:
        predictions: Predicted scores
        ground_truth: Ground truth relevance scores
        k: Number of items to consider
        
    Returns:
        NDCG@k score
    """
    # Get top-k items
    top_k_items = np.argsort(predictions)[-k:][::-1]
    
    # Calculate DCG
    dcg = 0
    for i, item in enumerate(top_k_items):
        if ground_truth[item] > 0:
            dcg += (2 ** ground_truth[item] - 1) / np.log2(i + 2)
    
    # Calculate IDCG
    ideal_ranking = np.sort(ground_truth)[-k:][::-1]
    idcg = 0
    for i, rel in enumerate(ideal_ranking):
        if rel > 0:
            idcg += (2 ** rel - 1) / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0


def calculate_hit_rate(predictions: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """
    Calculate Hit Rate@k
    
    Args:
        predictions: Predicted scores
        ground_truth: Ground truth relevance scores
        k: Number of items to consider
        
    Returns:
        Hit Rate@k score
    """
    # Get top-k items
    top_k_items = np.argsort(predictions)[-k:]
    
    # Check if any of the top-k items are relevant
    return 1.0 if np.any(ground_truth[top_k_items] > 0) else 0.0


def evaluate_model(
    model: BiVAE,
    data_loader: BiVAEDataLoader,
    k_values: List[int] = [5, 10, 20],
    device: torch.device = None
) -> Dict[str, float]:
    """
    Evaluate model performance using NDCG and Hit Rate
    
    Args:
        model: Trained BiVAE model
        data_loader: DataLoader instance
        k_values: List of k values for evaluation
        device: Device to use for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get test matrix
    test_matrix = data_loader.test_matrix
    train_matrix = data_loader.train_matrix
    
    # Initialize metrics
    metrics = {}
    for k in k_values:
        metrics[f'ndcg@{k}'] = []
        metrics[f'hr@{k}'] = []
    
    with torch.no_grad():
        # Get predictions for all users
        predictions = model.predict().cpu().numpy()
        
        # Evaluate for each user
        for user_idx in range(data_loader.num_users):
            # Get test items for this user
            test_items = test_matrix[user_idx].nonzero()[1]
            if len(test_items) == 0:
                continue
            
            # Get train items to exclude from recommendations
            train_items = train_matrix[user_idx].nonzero()[1]
            
            # Get user predictions and exclude training items
            user_preds = predictions[user_idx].copy()
            user_preds[train_items] = -np.inf
            
            # Get ground truth
            ground_truth = test_matrix[user_idx].toarray().flatten()
            
            # Calculate metrics for each k
            for k in k_values:
                ndcg = calculate_ndcg(user_preds, ground_truth, k)
                hr = calculate_hit_rate(user_preds, ground_truth, k)
                
                metrics[f'ndcg@{k}'].append(ndcg)
                metrics[f'hr@{k}'].append(hr)
    
    # Calculate average metrics
    results = {}
    for k in k_values:
        results[f'ndcg@{k}'] = np.mean(metrics[f'ndcg@{k}'])
        results[f'hr@{k}'] = np.mean(metrics[f'hr@{k}'])
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BiVAE model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to MovieLens data')
    
    # Evaluation parameters
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20],
                      help='K values for evaluation')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (auto, cpu, cuda)')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load data
    print("Loading data...")
    data_loader = BiVAEDataLoader(
        data_path=args.data_path,
        min_rating=4.0,
        test_size=0.2,
        random_state=42
    )
    
    # Create model
    print("Creating model...")
    model = BiVAE(
        num_users=data_loader.num_users,
        num_items=data_loader.num_items,
        k=50,  # Should match the trained model
        user_encoder_structure=[data_loader.num_items, 600, 200],
        item_encoder_structure=[data_loader.num_users, 600, 200],
        act_fn="tanh",
        likelihood="bern"
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = BiVAETrainer(
        model=model,
        data_loader=data_loader,
        learning_rate=0.001,
        kl_beta=1.0,
        batch_size=500,
        device=device
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    trainer.load_model(args.model_path)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(
        model=model,
        data_loader=data_loader,
        k_values=args.k_values,
        device=device
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for k in args.k_values:
        print(f"NDCG@{k}: {results[f'ndcg@{k}']:.4f}")
        print(f"HR@{k}: {results[f'hr@{k}']:.4f}")
        print("-" * 50)


if __name__ == '__main__':
    main() 