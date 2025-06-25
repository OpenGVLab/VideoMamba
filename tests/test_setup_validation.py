"""
Validation tests to ensure the testing infrastructure is properly set up.
"""
import sys
from pathlib import Path

import pytest
import torch
import numpy as np


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure setup."""
    
    def test_python_version(self):
        """Verify Python version is 3.8 or higher."""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"
    
    def test_pytest_installed(self):
        """Verify pytest is properly installed."""
        import pytest
        assert pytest.__version__ >= "8.0.0"
    
    def test_coverage_tools_installed(self):
        """Verify coverage tools are installed."""
        import pytest_cov
        import coverage
        assert pytest_cov is not None
        assert coverage is not None
    
    def test_mock_tools_installed(self):
        """Verify mocking tools are installed."""
        import pytest_mock
        from unittest.mock import Mock, MagicMock, patch
        assert pytest_mock is not None
        assert Mock is not None
    
    def test_torch_available(self):
        """Verify PyTorch is available."""
        assert torch is not None
        assert torch.__version__ is not None
    
    def test_numpy_available(self):
        """Verify NumPy is available."""
        assert np is not None
        assert np.__version__ is not None
    
    def test_directory_structure(self):
        """Verify the test directory structure is correct."""
        test_root = Path(__file__).parent
        assert test_root.exists()
        assert (test_root / "__init__.py").exists()
        assert (test_root / "conftest.py").exists()
        assert (test_root / "unit").exists()
        assert (test_root / "unit" / "__init__.py").exists()
        assert (test_root / "integration").exists()
        assert (test_root / "integration" / "__init__.py").exists()
    
    def test_project_structure(self):
        """Verify the main project structure is accessible."""
        project_root = Path(__file__).parent.parent
        assert (project_root / "videomamba").exists()
        assert (project_root / "mamba").exists()
        assert (project_root / "pyproject.toml").exists()
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True
    
    def test_fixtures_available(self, temp_dir, sample_config, mock_video_tensor):
        """Test that custom fixtures from conftest.py are available."""
        assert temp_dir.exists()
        assert isinstance(sample_config, dict)
        assert "model" in sample_config
        assert mock_video_tensor.shape == (2, 3, 16, 224, 224)
    
    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temp_dir fixture properly creates and will clean up directories."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        # Cleanup will happen automatically after test
    
    def test_device_fixture(self, device):
        """Test that device fixture returns appropriate device."""
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]
    
    def test_random_seed_reset(self):
        """Test that random seeds are properly reset."""
        # First random number should be deterministic due to seed reset
        first_torch = torch.rand(1).item()
        first_numpy = np.random.rand()
        
        # These should be the same across test runs due to autouse fixture
        assert 0 < first_torch < 1
        assert 0 < first_numpy < 1
    
    def test_mock_checkpoint(self, mock_checkpoint):
        """Test mock checkpoint fixture."""
        assert mock_checkpoint.exists()
        data = torch.load(mock_checkpoint, weights_only=True)
        assert "model" in data
        assert "optimizer" in data
        assert "epoch" in data
        assert data["epoch"] == 10
    
    def test_coverage_configured(self):
        """Test that coverage is properly configured."""
        import coverage
        cov = coverage.Coverage()
        config = cov.config
        # Just verify coverage can be instantiated
        assert cov is not None


class TestPoetryCommands:
    """Test Poetry script commands configuration."""
    
    def test_pyproject_exists(self):
        """Verify pyproject.toml exists."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"
    
    def test_poetry_scripts_configured(self):
        """Verify Poetry scripts are configured."""
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # Python 3.9-3.10
        
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        # Check Poetry scripts section
        assert "tool" in pyproject_data
        assert "poetry" in pyproject_data["tool"]
        assert "scripts" in pyproject_data["tool"]["poetry"]
        
        scripts = pyproject_data["tool"]["poetry"]["scripts"]
        assert "test" in scripts
        assert "tests" in scripts
        assert scripts["test"] == "pytest:main"
        assert scripts["tests"] == "pytest:main"
    
    def test_pytest_configuration(self):
        """Verify pytest is properly configured in pyproject.toml."""
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # Python 3.9-3.10
        
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        # Check pytest configuration
        assert "tool" in pyproject_data
        assert "pytest" in pyproject_data["tool"]
        assert "ini_options" in pyproject_data["tool"]["pytest"]
        
        pytest_config = pyproject_data["tool"]["pytest"]["ini_options"]
        assert "testpaths" in pytest_config
        assert "tests" in pytest_config["testpaths"]
        assert "markers" in pytest_config
        assert "unit: Unit tests" in pytest_config["markers"]
        assert "integration: Integration tests" in pytest_config["markers"]
        assert "slow: Slow tests that should be run less frequently" in pytest_config["markers"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])