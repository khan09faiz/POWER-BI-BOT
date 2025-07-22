"""
Qwen3 embedding model wrapper with GPU acceleration support
Optimized for RTX 3060 6GB VRAM with batch processing
"""
import torch
import numpy as np
from typing import List, Union, Optional, Tuple
import logging
from transformers import AutoModel, AutoTokenizer
import gc

from config import settings
from utils import log_execution_time, GPUMemoryMonitor, default_logger


class Qwen3Embedder:
    """Qwen3-Embedding-8B model wrapper with GPU optimization"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or default_logger
        self.model = None
        self.tokenizer = None
        self.device = None
        self.embedding_dim = None
        self.is_loaded = False

    @log_execution_time()
    def load_model(self):
        """Load Qwen3 model and tokenizer with optimized settings"""
        if self.is_loaded:
            self.logger.info("Model already loaded")
            return

        try:
            self.logger.info(f"Loading model: {settings.model.model_name}")

            # Determine device
            if settings.system.use_gpu and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{settings.system.cuda_device}")
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                self.logger.info("Using CPU")

            # Load tokenizer
            self.logger.info("Loading tokenizer.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.model.model_name,
                **settings.model.tokenizer_kwargs
            )

            # Load model with optimized settings
            self.logger.info("Loading model.")
            model_kwargs = settings.model.model_kwargs.copy()

            # Optimize for 6GB VRAM
            if self.device.type == "cuda":
                model_kwargs.update({
                    "torch_dtype": torch.float16,  # Use FP16 for memory efficiency
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                })

                # Clear GPU cache before loading
                torch.cuda.empty_cache()

            self.model = AutoModel.from_pretrained(
                settings.model.model_name,
                **model_kwargs
            )

            # Move to device if not already there
            if hasattr(self.model, 'to') and str(self.model.device) != str(self.device):
                self.model = self.model.to(self.device)

            # Set to evaluation mode
            self.model.eval()

            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size

            self.is_loaded = True
            self.logger.info(f"Model loaded successfully. Embedding dim: {self.embedding_dim}")

            # Log GPU memory usage
            if self.device.type == "cuda":
                GPUMemoryMonitor.log_memory_usage(self.logger, "Model Loading Complete")

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            self._cleanup()
            raise

    def _cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        self.is_loaded = False

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        """Prepare texts for embedding by combining fields"""
        prepared_texts = []

        for text in texts:
            if isinstance(text, str) and text.strip():
                # Clean and prepare text
                cleaned_text = text.strip().replace('\n', ' ').replace('\t', ' ')
                # Remove extra whitespace
                cleaned_text = ' '.join(cleaned_text.split())
                prepared_texts.append(cleaned_text)
            else:
                # Handle empty or null text
                prepared_texts.append("")

        return prepared_texts

    @log_execution_time()
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts to embeddings"""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")

        if not texts:
            return np.array([])

        # Prepare texts
        prepared_texts = self._prepare_texts(texts)

        try:
            with torch.no_grad():
                # Tokenize batch
                inputs = self.tokenizer(
                    prepared_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=settings.model.max_seq_length
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get embeddings
                outputs = self.model(**inputs)

                # Use mean pooling for sentence embeddings
                embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']

                # Apply attention mask and mean pool
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1).float()
                summed_embeddings = torch.sum(masked_embeddings, dim=1)
                summed_mask = torch.sum(attention_mask, dim=1, keepdim=True).float()
                mean_embeddings = summed_embeddings / torch.clamp(summed_mask, min=1e-9)

                # Normalize embeddings
                normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

                # Convert to numpy
                result = normalized_embeddings.cpu().numpy()

                return result

        except torch.cuda.OutOfMemoryError as e:
            self.logger.error("GPU out of memory. Try reducing batch size.")
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            self.logger.error(f"Error encoding batch: {str(e)}")
            raise

    @log_execution_time()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts with automatic batching"""
        if not texts:
            return np.array([])

        batch_size = settings.model.batch_size
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            self.logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            batch_embeddings = self.encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)

            # Clear GPU cache between batches
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Combine all embeddings
        if all_embeddings:
            result = np.vstack(all_embeddings)
            self.logger.info(f"Encoded {len(result)} texts into {result.shape[1]}-dimensional embeddings")
            return result
        else:
            return np.array([])

    def combine_equipment_fields(
        self,
        equipment_type: Optional[str] = None,
        equipment_description: Optional[str] = None
    ) -> str:
        """Combine equipment fields into a single text for embedding"""
        parts = []

        if equipment_type and str(equipment_type).strip() and str(equipment_type) != 'nan':
            parts.append(f"Type: {equipment_type}")

        if equipment_description and str(equipment_description).strip() and str(equipment_description) != 'nan':
            parts.append(f"Description: {equipment_description}")

        return " | ".join(parts) if parts else ""

    def __enter__(self):
        """Context manager entry"""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self._cleanup()
