"""
Dataset loading utilities.
"""
import os
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Tuple
import pandas as pd
import numpy as np

# Try to import fcntl (Unix only), fallback to no locking on Windows
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

from .config import (
    DATASET, BATCH_SIZE, TRAIN_SUBSET_SIZE, VAL_SUBSET_SIZE, DATA_ROOT,
    DATALOADER_NUM_WORKERS, TRAIN_DROP_LAST
)

# Dataset-dependent feature and class counts
if DATASET == "cifar10":
    IN_FEATURES: int = 32 * 32 * 3  # CIFAR-10: 32x32x3 RGB
    N_CLASSES: int = 10
elif DATASET == "svhn":
    IN_FEATURES: int = 32 * 32 * 3  # SVHN: 32x32x3 RGB (street view house numbers)
    N_CLASSES: int = 10             # SVHN: 10 classes (digits 0-9)
elif DATASET == "fashion-mnist":
    IN_FEATURES: int = 28 * 28      # Fashion-MNIST: 28x28x1 grayscale
    N_CLASSES: int = 10
elif DATASET == "emnist":
    IN_FEATURES: int = 28 * 28      # EMNIST: 28x28x1 grayscale
    N_CLASSES: int = 47             # EMNIST-balanced: 47 classes (digits + letters)
elif DATASET == "iris":
    IN_FEATURES: int = 4            # IRIS: 4 features
    N_CLASSES: int = 3              # IRIS: 3 classes
elif DATASET == "heart-failure" or DATASET == "heart_failure":
    IN_FEATURES: int = 11           # Heart Failure: 11 features (typical)
    N_CLASSES: int = 2              # Heart Failure: Binary classification
else:  # mnist (default)
    IN_FEATURES: int = 28 * 28      # MNIST: 28x28x1 grayscale
    N_CLASSES: int = 10

# Global cache for dataloaders (used in NSGA-II workers)
DATALOADERS: Optional[Tuple[DataLoader, DataLoader]] = None


def _get_dataset_path(dataset_name: str) -> str:
    """
    Get the dataset-specific folder path within DATA_ROOT.
    
    Maps dataset names to their organized folder structure:
    - cifar10 -> CIFAR10/
    - svhn -> SVHN/
    - fashion-mnist -> FashionMNIST/
    - mnist -> MNIST/
    - emnist -> EMNIST/
    - iris -> IRIS/ (for consistency, though iris doesn't download)
    - heart-failure -> HeartFailure/ (requires CSV file: heart_failure.csv)
    
    Args:
        dataset_name: Name of the dataset (e.g., "cifar10", "fashion-mnist")
        
    Returns:
        Path to the dataset folder (e.g., "data/CIFAR10")
    """
    # Map dataset names to folder names
    folder_map = {
        "cifar10": "CIFAR10",
        "svhn": "SVHN",
        "fashion-mnist": "FashionMNIST",
        "mnist": "MNIST",
        "emnist": "EMNIST",
        "iris": "IRIS",
        "heart-failure": "HeartFailure",
        "heart_failure": "HeartFailure"
    }
    
    folder_name = folder_map.get(dataset_name.lower(), dataset_name.upper())
    return os.path.join(DATA_ROOT, folder_name)


def _download_dataset_with_lock(dataset_class, dataset_path, **kwargs):
    """
    Download dataset with file locking to prevent corruption from concurrent downloads.
    
    Args:
        dataset_class: The dataset class (e.g., datasets.FashionMNIST)
        dataset_path: Path where dataset should be stored
        **kwargs: Additional arguments to pass to dataset_class
    
    Returns:
        The dataset instance
    """
    max_retries = 5
    retry_delay = 2.0
    
    # On Windows or systems without fcntl, use simple retry logic without file locking
    if not HAS_FCNTL:
        for attempt in range(max_retries):
            try:
                return dataset_class(root=dataset_path, download=True, **kwargs)
            except (OSError, IOError, RuntimeError, EOFError) as e:
                error_msg = str(e)
                if attempt < max_retries - 1 and ("corrupted" in error_msg.lower() or 
                                                   "eof" in error_msg.lower() or 
                                                   "file not found" in error_msg.lower()):
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
        return dataset_class(root=dataset_path, download=True, **kwargs)
    
    # Unix systems: use file locking
    lock_file = Path(dataset_path) / ".download_lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            # Try to acquire lock and download
            with open(lock_file, 'w') as f:
                try:
                    # Try to acquire exclusive lock (non-blocking)
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # Lock acquired, download dataset
                    dataset = dataset_class(root=dataset_path, download=True, **kwargs)
                    
                    # Release lock by closing file
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return dataset
                    
                except BlockingIOError:
                    # Lock is held by another process, wait and retry
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Block until lock is available
                    # Lock acquired, try to load dataset (might already be downloaded)
                    try:
                        dataset = dataset_class(root=dataset_path, download=False, **kwargs)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        return dataset
                    except Exception:
                        # Dataset not fully downloaded, download it
                        dataset = dataset_class(root=dataset_path, download=True, **kwargs)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        return dataset
                        
        except (OSError, IOError, RuntimeError, EOFError) as e:
            # Handle file corruption errors
            if attempt < max_retries - 1:
                error_msg = str(e)
                if "corrupted" in error_msg.lower() or "eof" in error_msg.lower() or "file not found" in error_msg.lower():
                    # Corrupted file detected, wait and retry
                    time.sleep(retry_delay * (attempt + 1))
                    # Try to remove corrupted files if they exist
                    try:
                        dataset_path_obj = Path(dataset_path)
                        if dataset_path_obj.exists():
                            # Remove potentially corrupted files
                            for file in dataset_path_obj.rglob("*"):
                                if file.is_file() and not file.name.endswith('.lock'):
                                    try:
                                        file.unlink()
                                    except:
                                        pass
                    except:
                        pass
                    continue
            # Re-raise if max retries reached or non-corruption error
            raise
        except Exception as e:
            # For other errors, try without lock (fallback)
            if attempt == max_retries - 1:
                return dataset_class(root=dataset_path, download=True, **kwargs)
            time.sleep(retry_delay)
            continue
    
    # Final fallback
    return dataset_class(root=dataset_path, download=True, **kwargs)


def make_subset(ds, n, seed=35):
    """Create a random subset of a dataset."""
    if n <= 0 or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    return Subset(ds, idx.tolist())


def _load_heart_failure_dataset(dataset_path: str, train_ratio: float = 0.8, seed: int = 35) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """
    Load Heart Failure Prediction Dataset from CSV file.
    
    Expected CSV format:
    - Features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, 
                RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
    - Target: HeartDisease (or similar binary column)
    
    The function will:
    1. Look for heart_failure.csv or heart-failure.csv in the dataset path
    2. Handle both numeric and categorical features
    3. Normalize features using StandardScaler
    4. Encode categorical variables
    5. Split into train/test sets
    
    Args:
        dataset_path: Path to the dataset folder
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, test_dataset) as TensorDataset
    """
    dataset_path_obj = Path(dataset_path)
    dataset_path_obj.mkdir(parents=True, exist_ok=True)
    
    # Try to find the CSV file
    csv_files = [
        dataset_path_obj / "heart_failure.csv",
        dataset_path_obj / "heart-failure.csv",
        dataset_path_obj / "heart_failure_prediction.csv",
        dataset_path_obj / "heart.csv"
    ]
    
    csv_file = None
    for f in csv_files:
        if f.exists():
            csv_file = f
            break
    
    if csv_file is None:
        raise FileNotFoundError(
            f"Heart Failure dataset CSV not found. Please place a CSV file named one of: "
            f"{[f.name for f in csv_files]} in {dataset_path}. "
            f"Expected columns: Age, Sex, ChestPainType, RestingBP, Cholesterol, "
            f"FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease"
        )

    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Identify target column (common names)
    target_candidates = ['HeartDisease', 'heart_disease', 'HeartDisease', 'target', 'Target', 'label', 'Label']
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Assume last column is target
        target_col = df.columns[-1]
        print(f"[WARN] Target column not found, using last column: {target_col}")
    
    # Separate features and target
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])
    
    # Handle categorical columns
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = X_df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_df[col].astype(str))
        label_encoders[col] = le
    
    # Convert to numpy array
    X = X_encoded.values.astype(np.float32)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Ensure binary classification (0 and 1)
    if y_tensor.max() > 1:
        y_tensor = (y_tensor > y_tensor.median()).long()
    elif y_tensor.min() > 0:
        y_tensor = y_tensor - y_tensor.min()
    
    # Split into train/test
    n_samples = len(X_tensor)
    n_train = int(train_ratio * n_samples)
    
    # Use seed for reproducible split
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=g)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_dataset = torch.utils.data.TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
    test_dataset = torch.utils.data.TensorDataset(X_tensor[test_indices], y_tensor[test_indices])
    
    print(f"[INFO] Loaded Heart Failure dataset: {n_samples} samples, {X_tensor.shape[1]} features, "
          f"{len(train_dataset)} train, {len(test_dataset)} test")
    
    return train_dataset, test_dataset


def _load_iris_dataset(train_ratio: float = 0.8, seed: int = 35) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """
    Load Iris dataset from sklearn, normalize features, and split into train/test.

    Args:
        train_ratio: Fraction of data for training (default: 0.8).
        seed: Random seed for reproducible split.

    Returns:
        Tuple of (train_dataset, test_dataset) as TensorDataset.
    """
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    n_samples = len(X_tensor)
    n_train = int(train_ratio * n_samples)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=g)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_dataset = torch.utils.data.TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
    test_dataset = torch.utils.data.TensorDataset(X_tensor[test_indices], y_tensor[test_indices])

    print(f"[INFO] Loaded Iris dataset: {n_samples} samples, 4 features, 3 classes, "
          f"{len(train_dataset)} train, {len(test_dataset)} test")
    return train_dataset, test_dataset


def get_dataloaders(in_pool_worker: bool, train_size: Optional[int] = None, 
                   val_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create (or return cached) DataLoaders for this process.
    
    Args:
        in_pool_worker: Whether running in a multiprocessing pool worker
        train_size: Optional override for training subset size
        val_size: Optional override for validation subset size
    """
    global DATALOADERS
    
    actual_train_size = train_size if train_size is not None else TRAIN_SUBSET_SIZE
    actual_val_size = val_size if val_size is not None else VAL_SUBSET_SIZE
    
    # Use cache if sizes match defaults
    use_cache = (train_size is None and val_size is None and DATALOADERS is not None)
    if use_cache:
        return DATALOADERS

    # Get dataset-specific path
    dataset_path = _get_dataset_path(DATASET)
    
    # Load dataset based on DATASET config
    # Use locked download to prevent corruption from concurrent downloads
    if DATASET == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_ds = _download_dataset_with_lock(datasets.CIFAR10, dataset_path, train=True, transform=transform)
        test_ds = _download_dataset_with_lock(datasets.CIFAR10, dataset_path, train=False, transform=transform)
    elif DATASET == "svhn":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        train_ds = _download_dataset_with_lock(datasets.SVHN, dataset_path, split='train', transform=transform)
        test_ds = _download_dataset_with_lock(datasets.SVHN, dataset_path, split='test', transform=transform)
    elif DATASET == "fashion-mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_ds = _download_dataset_with_lock(datasets.FashionMNIST, dataset_path, train=True, transform=transform)
        test_ds = _download_dataset_with_lock(datasets.FashionMNIST, dataset_path, train=False, transform=transform)
    elif DATASET == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,))
        ])
        train_ds = _download_dataset_with_lock(datasets.EMNIST, dataset_path, split='balanced', train=True, transform=transform)
        test_ds = _download_dataset_with_lock(datasets.EMNIST, dataset_path, split='balanced', train=False, transform=transform)
    elif DATASET == "iris":
        train_ds, test_ds = _load_iris_dataset(train_ratio=0.8, seed=35)
    elif DATASET == "heart-failure" or DATASET == "heart_failure":
        train_ds, test_ds = _load_heart_failure_dataset(dataset_path, train_ratio=0.8, seed=35)
    else:  # mnist (default)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = _download_dataset_with_lock(datasets.MNIST, dataset_path, train=True, transform=transform)
        test_ds = _download_dataset_with_lock(datasets.MNIST, dataset_path, train=False, transform=transform)

    train_sub = make_subset(train_ds, actual_train_size)
    val_sub = make_subset(test_ds, actual_val_size)

    # DataLoader configuration
    if in_pool_worker:
        num_workers = 0  # Must be 0 in pool workers
    else:
        num_workers = DATALOADER_NUM_WORKERS
    pin_mem = torch.cuda.is_available()
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, drop_last=TRAIN_DROP_LAST,
                              num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=pin_mem)
    
    # Cache if using default sizes
    if train_size is None and val_size is None:
        DATALOADERS = (train_loader, val_loader)
    
    return (train_loader, val_loader)









