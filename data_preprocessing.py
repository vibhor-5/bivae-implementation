import polars as pl
import torch
import random

def load_interaction_tensor(csv_path: str, threshold: float = 0.0, binary: bool = True):
    """
    Loads ratings.csv using Polars and converts to a user-item interaction matrix tensor.

    Args:
        csv_path (str): Path to ratings.csv. Assumes columns: userId, movieId, rating
        threshold (float): Minimum rating value to count as an interaction if binary=True.
        binary (bool): If True, creates binary interaction matrix, else uses rating values.

    Returns:
        interaction_tensor (torch.Tensor): Shape [n_users, n_items]
        user_id_map (dict): Maps original userId to row index
        item_id_map (dict): Maps original movieId to column index
    """
    # Load with Polars
    with open(csv_path, "r") as f:
        rows = [line.strip().split("::") for line in f]

    # Convert to Polars DataFrame with appropriate column names and types
    df = pl.DataFrame(rows, schema=["userId", "movieId", "rating", "timestamp"]).with_columns([
        pl.col("userId").cast(pl.Int32),
        pl.col("movieId").cast(pl.Int32),
        pl.col("rating").cast(pl.Float32),
        pl.col("timestamp").cast(pl.Int64)
    ])

    print(df.head())

    # Get unique users/items and build ID maps
    user_ids = df.select("userId").unique().sort("userId").to_series().to_list()
    item_ids = df.select("movieId").unique().sort("movieId").to_series().to_list()
    print(f"Number of unique users: {len(user_ids)}")
    print(f"Number of unique items: {len(item_ids)}")

    user_id_map = {uid: i for i, uid in enumerate(user_ids)}
    item_id_map = {iid: i for i, iid in enumerate(item_ids)}

    # Initialize interaction matrix
    n_users = len(user_ids)
    n_items = len(item_ids)
    matrix = torch.zeros((n_users, n_items), dtype=torch.float32)

    # Populate matrix
    for row in df.iter_rows(named=True):
        u_idx = user_id_map[row["userId"]]
        i_idx = item_id_map[row["movieId"]]
        rating = row["rating"]
        if binary:
            matrix[u_idx, i_idx] = float(rating >= threshold)
        else:
            matrix[u_idx, i_idx] = float(rating)

    return matrix, user_id_map, item_id_map

def split_user_interactions(matrix: torch.Tensor, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    Splits user-item interaction matrix into train/val/test sets in 70/10/20 ratio per user.

    Args:
        matrix (torch.Tensor): Binary or real-valued interaction matrix [n_users, n_items]
        val_ratio (float): Ratio of interactions per user to go to validation
        test_ratio (float): Ratio of interactions per user to go to test
        seed (int): Random seed for reproducibility

    Returns:
        train_matrix (torch.Tensor)
        val_matrix (torch.Tensor)
        test_matrix (torch.Tensor)
    """
    torch.manual_seed(seed)
    random.seed(seed)

    n_users, n_items = matrix.shape
    train = torch.zeros_like(matrix)
    val = torch.zeros_like(matrix)
    test = torch.zeros_like(matrix)

    for user in range(n_users):
        item_indices = matrix[user].nonzero(as_tuple=False).squeeze().tolist()
        if isinstance(item_indices, int):
            item_indices = [item_indices]
        num_items = len(item_indices)

        if num_items == 0:
            continue  # Skip users with no interactions

        random.shuffle(item_indices)

        num_val = max(1, int(val_ratio * num_items))
        num_test = max(1, int(test_ratio * num_items))
        num_train = num_items - num_val - num_test

        train_items = item_indices[:num_train]
        val_items = item_indices[num_train:num_train + num_val]
        test_items = item_indices[num_train + num_val:]

        train[user, train_items] = matrix[user, train_items]
        val[user, val_items] = matrix[user, val_items]
        test[user, test_items] = matrix[user, test_items]

    return train, val, test
