"""
Semantic search using FAISS for efficient similarity matching
Supports GPU acceleration and various index types
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import logging

try:
    import faiss
except ImportError:
    faiss = None

from config import settings
from utils import log_execution_time, GPUMemoryMonitor, default_logger


class FAISSSearchEngine:
    """FAISS-based semantic search engine with GPU support"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or default_logger
        self.index = None
        self.master_data = None
        self.embeddings = None
        self.dimension = None
        self.is_gpu_index = False

        if faiss is None:
            self.logger.warning("FAISS not installed. Install with: pip install faiss-gpu faiss-cpu")

    def _check_faiss(self):
        """Check if FAISS is available"""
        if faiss is None:
            raise ImportError("FAISS not installed. Install with: pip install faiss-gpu faiss-cpu")

    def _create_index(self, dimension: int):
        """Create FAISS index based on configuration"""
        self._check_faiss()

        index_type = settings.search.index_type.lower()

        if index_type == "flat":
            # Flat (exact) index - best quality but slower for large datasets
            index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        elif index_type == "ivfflat":
            # IVFFlat index - good balance of speed and accuracy
            nlist = min(settings.search.nlist, len(self.embeddings) // 10)  # Avoid too many clusters
            nlist = max(nlist, 1)  # Ensure at least 1 cluster

            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        elif index_type == "hnsw":
            # HNSW index - very fast approximate search
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is M parameter
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16

        else:
            self.logger.warning(f"Unknown index type: {index_type}, using Flat")
            index = faiss.IndexFlatIP(dimension)

        return index

    def _move_index_to_gpu(self, index):
        """Move FAISS index to GPU if available and requested"""
        if not settings.search.use_gpu_index:
            return index

        if not faiss.get_num_gpus() > 0:
            self.logger.warning("GPU requested for FAISS but no GPUs available")
            return index

        try:
            # Create GPU resources
            gpu_res = faiss.StandardGpuResources()

            # Configure GPU memory
            if settings.system.use_gpu:
                # Use a portion of GPU memory for FAISS
                available_memory = int(6 * 1024**3 * settings.system.max_memory_usage * 0.3)  # 30% of available
                gpu_res.setDefaultNullStreamAllDeviceMemory(available_memory)

            # Move to GPU
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, settings.system.cuda_device or 0, index)

            self.is_gpu_index = True
            self.logger.info("FAISS index moved to GPU")

            if settings.system.use_gpu:
                GPUMemoryMonitor.log_memory_usage(self.logger, "FAISS GPU Index Created")

            return gpu_index

        except Exception as e:
            self.logger.warning(f"Failed to move FAISS index to GPU: {str(e)}")
            return index

    @log_execution_time()
    def build_index(self, embeddings: np.ndarray, master_data: pd.DataFrame):
        """Build FAISS index from embeddings and master data"""
        self._check_faiss()

        if len(embeddings) == 0:
            raise ValueError("No embeddings provided for index building")

        if len(embeddings) != len(master_data):
            raise ValueError("Embeddings and master data length mismatch")

        self.embeddings = embeddings.astype(np.float32)
        self.master_data = master_data.copy()
        self.dimension = embeddings.shape[1]

        self.logger.info(f"Building FAISS index: {len(embeddings)} vectors, {self.dimension} dimensions")

        # Create index
        self.index = self._create_index(self.dimension)

        # Move to GPU if requested and available
        self.index = self._move_index_to_gpu(self.index)

        # Train index if needed (for IVF types)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.logger.info("Training FAISS index...")
            self.index.train(self.embeddings)

        # Add vectors to index
        self.logger.info("Adding vectors to index...")
        self.index.add(self.embeddings)

        self.logger.info(f"Index built successfully. Total vectors: {self.index.ntotal}")

    @log_execution_time()
    def search_similar(
        self,
        query_embeddings: np.ndarray,
        k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[List[Tuple[int, float]]]:
        """Search for similar vectors in the index"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if len(query_embeddings) == 0:
            return []

        # Use config defaults if not provided
        k = k or settings.search.top_k_candidates
        threshold = threshold or settings.search.similarity_threshold

        query_embeddings = query_embeddings.astype(np.float32)

        # Search
        scores, indices = self.index.search(query_embeddings, k)

        # Convert to list of tuples with filtering by threshold
        results = []
        for i in range(len(query_embeddings)):
            query_results = []
            for j in range(k):
                idx = indices[i][j]
                score = scores[i][j]

                # Filter by threshold and valid indices
                if idx != -1 and score >= threshold:
                    query_results.append((idx, float(score)))

            results.append(query_results)

        return results

    def get_master_record(self, index: int) -> Dict[str, Any]:
        """Get master record by index"""
        if self.master_data is None or index >= len(self.master_data):
            return {}

        record = self.master_data.iloc[index].to_dict()
        return record

    def find_best_matches(
        self,
        query_embeddings: np.ndarray,
        missing_fields: List[str]
    ) -> List[Dict[str, Any]]:
        """Find best matches and extract requested fields"""
        search_results = self.search_similar(query_embeddings)

        matches = []
        for i, query_results in enumerate(search_results):
            match_info = {
                'query_index': i,
                'found_matches': len(query_results) > 0,
                'best_score': 0.0,
                'filled_fields': {}
            }

            if query_results:
                # Get best match (highest score)
                best_idx, best_score = max(query_results, key=lambda x: x[1])
                match_info['best_score'] = best_score

                # Get master record
                master_record = self.get_master_record(best_idx)

                # Extract requested fields
                for field in missing_fields:
                    if field in master_record and pd.notna(master_record[field]):
                        match_info['filled_fields'][field] = master_record[field]

                # Add similarity score for each field
                for field in missing_fields:
                    if field in match_info['filled_fields']:
                        match_info['filled_fields'][f'{field}_similarity_score'] = best_score

            matches.append(match_info)

        return matches

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        if self.index is None:
            return {"status": "not_built"}

        stats = {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "is_trained": getattr(self.index, 'is_trained', True),
            "is_gpu": self.is_gpu_index
        }

        return stats

    def cleanup(self):
        """Clean up FAISS resources"""
        if self.index is not None:
            del self.index
            self.index = None

        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None

        self.master_data = None
        self.is_gpu_index = False
