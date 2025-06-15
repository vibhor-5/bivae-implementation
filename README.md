# BiVAE Implementation

This project implements a Bilateral Variational Autoencoder (BiVAE) for collaborative filtering, specifically designed for recommendation systems. The implementation is based on the paper "Bilateral Variational Autoencoder for Collaborative Filtering" and provides a modular, efficient, and scalable solution for recommendation tasks.

## Project Structure

```
.
├── bivae_model.py          # Core BiVAE model implementation
├── bivae_trainer.py        # Training and evaluation logic
├── data_loader.py          #Data loading and preprocessing
├── train.py                # Model Training script
├── test.py                 # Model evaluation script
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Features

- **Modular Architecture**: Clean separation of model, training, and data loading components
- **Bilateral VAE**: Implements both user and item encoders with shared latent space
- **Flexible Training**: Supports various training configurations and hyperparameters
- **Evaluation Metrics**: Implements NDCG and Hit Rate for model evaluation
- **GPU Support**: Efficient training with CUDA support
- **Checkpointing**: Model checkpointing with best model saving

## Requirements

- Python 3.10+
- PyTorch 2.6+
- NumPy
- SciPy
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Architecture

The BiVAE model consists of:

1. **User Encoder**:

   - Input: User-item interaction matrix
   - Architecture: [num_items, 600, 200] → latent space
   - Activation: tanh

2. **Item Encoder**:

   - Input: Item-user interaction matrix
   - Architecture: [num_users, 600, 200] → latent space
   - Activation: tanh

3. **Shared Latent Space**:

   - Dimension: 50 (configurable)
   - Gaussian distribution

4. **Decoders**:
   - Bernoulli likelihood for binary interactions
   - Reconstruction of original interaction matrices

## Training

The training process includes:

1. **Data Preprocessing**:

   - Binarization of ratings (default threshold: 4.0)
   - Train/test split (default: 80/20)
   - Matrix conversion to sparse format

2. **Training Loop**:

   - Alternating updates of user and item encoders
   - KL divergence regularization
   - Learning rate scheduling
   - Early stopping based on NDCG

3. **Optimization**:
   - Adam optimizer
   - Separate optimizers for user and item encoders
   - Learning rate scheduling with cosine annealing

## Evaluation

The model is evaluated using:

1. **NDCG@k** (Normalized Discounted Cumulative Gain):

   - Measures ranking quality
   - Default k values: [5, 10, 20]

2. **Hit Rate@k**:
   - Measures recommendation accuracy
   - Default k values: [5, 10, 20]

## Usage

### Training

```python
from bivae_model import BiVAE
from bivae_trainer import BiVAETrainer
from data_loader import BiVAEDataLoader

# Load data
data_loader = BiVAEDataLoader(
    data_path="path/to/movielens/data",
    min_rating=4.0,
    test_size=0.2
)

# Create model
model = BiVAE(
    num_users=data_loader.num_users,
    num_items=data_loader.num_items,
    k=50,
    user_encoder_structure=[data_loader.num_items, 600, 200],
    item_encoder_structure=[data_loader.num_users, 600, 200],
    act_fn="tanh",
    likelihood="bern"
)

# Create trainer
trainer = BiVAETrainer(
    model=model,
    data_loader=data_loader,
    learning_rate=0.001,
    kl_beta=1.0,
    batch_size=500
)

# Train model
trainer.train(epochs=100)
```

### Evaluation

```bash
python test.py --model_path path/to/model/checkpoint.pt --data_path path/to/movielens/data
```

## Model Checkpoints

The trainer saves checkpoints in the following format:

- `best_model.pt`: Best model based on validation NDCG
- `checkpoint_epoch_{N}.pt`: Checkpoint at epoch N

Checkpoints contain:

- Model state dict
- Optimizer state dicts
- Training history
- Current epoch
- Best NDCG score

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- Based on the paper "Bilateral Variational Autoencoder for Collaborative Filtering"
- Uses MovieLens dataset for evaluation
- Built with PyTorch
