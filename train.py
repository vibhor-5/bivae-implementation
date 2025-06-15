import argparse
import torch
from data_loader import BiVAEDataLoader
from bivae_model import BiVAE
from bivae_trainer import BiVAETrainer
import os
from datetime import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train BiVAE model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to MovieLens data')
    parser.add_argument('--min_rating', type=float, default=4.0,
                      help='Minimum rating for implicit feedback')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Test set ratio')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=50,
                      help='Latent dimension')
    parser.add_argument('--user_encoder_structure', type=int, nargs='+',
                      default=[600, 200],
                      help='User encoder structure')
    parser.add_argument('--item_encoder_structure', type=int, nargs='+',
                      default=[600, 200],
                      help='Item encoder structure')
    parser.add_argument('--act_fn', type=str, default='tanh',
                      choices=['sigmoid', 'tanh', 'relu', 'elu'],
                      help='Activation function')
    parser.add_argument('--likelihood', type=str, default='bern',
                      choices=['bern', 'gaus', 'pois'],
                      help='Likelihood function')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=500,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--kl_beta', type=float, default=1.0,
                      help='KL divergence weight')
    parser.add_argument('--eval_every', type=int, default=10,
                      help='Evaluate every N epochs')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--project_name', type=str, default='BiVAE',
                      help='Project name for logging')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_loader = BiVAEDataLoader(
        data_path=args.data_path,
        min_rating=args.min_rating,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    # Create model
    print("Creating model...")
    model = BiVAE(
        num_users=data_loader.num_users,
        num_items=data_loader.num_items,
        k=args.latent_dim,
        user_encoder_structure=[data_loader.num_items] + args.user_encoder_structure,
        item_encoder_structure=[data_loader.num_users] + args.item_encoder_structure,
        act_fn=args.act_fn,
        likelihood=args.likelihood
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = BiVAETrainer(
        model=model,
        data_loader=data_loader,
        learning_rate=args.lr,
        kl_beta=args.kl_beta,
        batch_size=args.batch_size,
        device=device,
        checkpoint_dir=checkpoint_dir,
        project_name=args.project_name
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        num_epochs=args.epochs,
        eval_every=args.eval_every,
        verbose=True,
        save_path=os.path.join(checkpoint_dir, 'best_model.pt')
    )
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(checkpoint_dir, 'training_history.png')
    )
    
    print("Training completed!")
    print(f"Checkpoints saved in: {checkpoint_dir}")


if __name__ == '__main__':
    main()


        


