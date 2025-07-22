"""
Configuration settings for semantic data completion pipeline
Handles GPU/CPU settings, model configuration, and runtime parameters
"""
import os
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for Qwen3 embedding model"""
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    device: str = "auto"  # "auto", "cuda", "cpu"
    max_seq_length: int = 512
    batch_size: int = 16
    trust_remote_code: bool = True

    # Model-specific kwargs
    model_kwargs: Dict[str, Any] = None
    tokenizer_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
            }

        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {
                "trust_remote_code": True,
                "padding": True,
                "truncation": True,
                "max_length": self.max_seq_length,
            }


@dataclass
class SearchConfig:
    """Configuration for FAISS semantic search"""
    similarity_threshold: float = 0.9
    top_k_candidates: int = 5
    use_gpu_index: bool = True
    index_type: str = "IVFFlat"
    nlist: int = 100


@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline"""
    supported_formats: tuple = (".csv", ".xlsx", ".xls")
    chunk_size: int = 1000
    include_similarity_scores: bool = True
    output_format: str = "xlsx"
    backup_original: bool = True


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    use_gpu: bool = True
    cuda_device: Optional[int] = None
    max_memory_usage: float = 0.7
    enable_mixed_precision: bool = True
    num_workers: int = 2

    def __post_init__(self):
        # Auto-detect CUDA availability
        if self.use_gpu and not torch.cuda.is_available():
            print("Warning: GPU requested but CUDA not available. Falling back to CPU.")
            self.use_gpu = False

        # Set CUDA device if available
        if self.use_gpu and torch.cuda.is_available():
            if self.cuda_device is None:
                self.cuda_device = 0
            torch.cuda.set_device(self.cuda_device)


class Settings:
    """Main settings class combining all configurations"""

    def __init__(
        self,
        use_gpu: bool = True,
        similarity_threshold: float = 0.75,
        batch_size: Optional[int] = None,
        custom_model_name: Optional[str] = None
    ):
        # Initialize system config first
        self.system = SystemConfig(use_gpu=use_gpu)

        if batch_size is None:
            if self.system.use_gpu and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                batch_size = 32 if gpu_memory > 8 else 16
            else:
                batch_size = 8

        # Initialize model config
        self.model = ModelConfig(
            model_name=custom_model_name or "Qwen/Qwen3-Embedding-8B",
            device="cuda" if self.system.use_gpu else "cpu",
            batch_size=batch_size
        )

        # Initialize search config
        self.search = SearchConfig(
            similarity_threshold=similarity_threshold,
            use_gpu_index=self.system.use_gpu
        )

        # Initialize processing config
        self.processing = ProcessingConfig()

    def get_device_info(self) -> Dict[str, Any]:
        """Get current device information"""
        info = {
            "use_gpu": self.system.use_gpu,
            "cuda_available": torch.cuda.is_available(),
            "device": self.model.device,
            "batch_size": self.model.batch_size,
        }

        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "gpu_count": torch.cuda.device_count(),
            })

        return info

    def validate(self) -> bool:
        """Validate current configuration"""
        if self.system.use_gpu and not torch.cuda.is_available():
            return False

        if self.search.similarity_threshold < 0 or self.search.similarity_threshold > 1:
            return False

        if self.model.batch_size <= 0:
            return False

        return True

settings = Settings()
