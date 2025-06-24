"""
Shared pytest fixtures and configuration for VideoMamba tests.
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Generator, Any

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that's cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration dictionary for testing."""
    return {
        "model": {
            "name": "videomamba_tiny",
            "num_classes": 400,
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "head_drop_rate": 0.0,
        },
        "data": {
            "dataset": "kinetics400",
            "num_frames": 16,
            "sampling_rate": 4,
            "num_workers": 8,
            "batch_size": 32,
        },
        "optimization": {
            "lr": 1e-4,
            "min_lr": 1e-6,
            "warmup_epochs": 5,
            "epochs": 30,
            "weight_decay": 0.05,
        },
    }


@pytest.fixture
def omega_config(sample_config: Dict[str, Any]) -> OmegaConf:
    """Create an OmegaConf configuration object."""
    return OmegaConf.create(sample_config)


@pytest.fixture
def mock_video_tensor() -> torch.Tensor:
    """Create a mock video tensor for testing (B, C, T, H, W)."""
    batch_size = 2
    channels = 3
    num_frames = 16
    height = 224
    width = 224
    return torch.randn(batch_size, channels, num_frames, height, width)


@pytest.fixture
def mock_image_tensor() -> torch.Tensor:
    """Create a mock image tensor for testing (B, C, H, W)."""
    batch_size = 4
    channels = 3
    height = 224
    width = 224
    return torch.randn(batch_size, channels, height, width)


@pytest.fixture
def mock_labels() -> torch.Tensor:
    """Create mock classification labels."""
    batch_size = 4
    num_classes = 400
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_checkpoint(temp_dir: Path) -> Path:
    """Create a mock checkpoint file."""
    checkpoint_path = temp_dir / "mock_checkpoint.pth"
    checkpoint_data = {
        "model": {"dummy_weight": torch.randn(10, 10)},
        "optimizer": {"lr": 0.001},
        "epoch": 10,
        "best_acc": 0.85,
    }
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def mock_video_file(temp_dir: Path) -> Path:
    """Create a mock video file path (doesn't create actual video)."""
    video_path = temp_dir / "test_video.mp4"
    video_path.touch()  # Create empty file
    return video_path


@pytest.fixture
def mock_dataset_root(temp_dir: Path) -> Path:
    """Create a mock dataset directory structure."""
    dataset_root = temp_dir / "mock_dataset"
    dataset_root.mkdir()
    
    # Create some mock class directories
    for i in range(5):
        class_dir = dataset_root / f"class_{i:03d}"
        class_dir.mkdir()
        
        # Create some mock video files
        for j in range(3):
            video_file = class_dir / f"video_{j:03d}.mp4"
            video_file.touch()
    
    return dataset_root


@pytest.fixture
def mock_transforms():
    """Provide mock transform configuration."""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def cleanup_files():
    """Fixture to track and cleanup files created during tests."""
    files_to_cleanup = []
    
    def add_file(filepath: Path):
        files_to_cleanup.append(filepath)
    
    yield add_file
    
    # Cleanup
    for filepath in files_to_cleanup:
        if filepath.exists():
            if filepath.is_file():
                filepath.unlink()
            elif filepath.is_dir():
                shutil.rmtree(filepath)


@pytest.fixture
def monkeypatch_env(monkeypatch):
    """Fixture to safely set environment variables during tests."""
    def setenv(key: str, value: str):
        monkeypatch.setenv(key, value)
    
    return setenv


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and assert on log messages."""
    with caplog.at_level("DEBUG"):
        yield caplog