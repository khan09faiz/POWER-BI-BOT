# Semantic Data Completion Pipeline

A modular, GPU-accelerated semantic data completion system using Qwen3-Embedding-8B for filling missing equipment metadata fields through intelligent similarity matching.

## 🚀 Features

- **GPU Acceleration**: Optimized for RTX 3060 6GB with CUDA 12.9 support
- **Qwen3 Embeddings**: Uses state-of-the-art Qwen/Qwen3-Embedding-8B model
- **FAISS Search**: High-performance similarity search with GPU acceleration
- **Multi-Format Support**: CSV, XLS, XLSX file compatibility
- **Modular Design**: Clean separation of concerns for easy extension
- **Robust Error Handling**: Comprehensive validation and logging
- **Similarity Scoring**: Optional similarity confidence scores in output
- **Memory Optimized**: Efficient batch processing for limited VRAM

## 📋 Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX 3060 6GB)
- CUDA 12.9 or compatible version
- Minimum 8GB system RAM recommended

### Software
- Python 3.8+
- PyTorch with CUDA support
- FAISS (CPU and GPU versions)
- Transformers library

## 🛠️ Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Verify GPU Setup**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'FAISS GPU Support: {faiss.get_num_gpus() > 0}')"
```

## 📁 Project Structure

```
src/embeddings/
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration management
├── data_processing/
│   ├── __init__.py
│   ├── data_loader.py        # Multi-format data loading
│   └── data_writer.py        # Output generation
├── model/
│   ├── __init__.py
│   └── qwen_embedder.py      # Qwen3 embedding wrapper
├── search/
│   ├── __init__.py
│   └── semantic_search.py    # FAISS similarity search
├── pipeline/
│   ├── __init__.py
│   └── filler.py            # Main orchestration logic
├── utils/
│   ├── __init__.py
│   └── logger.py            # Logging and monitoring
├── __init__.py
└── embedding.py             # CLI entry point
```

## 🎯 Usage

### Command Line Interface

```bash
# Basic usage with GPU acceleration
python src/embeddings/embedding.py master_data.xlsx target_data.xlsx

# Custom similarity threshold and output file
python src/embeddings/embedding.py master.csv target.csv -o completed.xlsx -t 0.8

# Force CPU usage with custom batch size
python src/embeddings/embedding.py master.xlsx target.xlsx --cpu --batch-size 8

# Show device information
python src/embeddings/embedding.py --info
```

### Python API

```python
from src.embeddings import SemanticDataFiller

# Using context manager for automatic cleanup
with SemanticDataFiller() as pipeline:
    results = pipeline.process_files(
        master_file="master_data.xlsx",
        target_file="target_data.xlsx",
        output_file="completed_data.xlsx"
    )

print(f"Status: {results['status']}")
print(f"Output: {results['output_file']}")
```

### Configuration Options

```python
from src.embeddings.config import Settings

# Custom configuration
settings = Settings(
    use_gpu=True,
    similarity_threshold=0.8,
    batch_size=32,
    custom_model_name="Qwen/Qwen3-Embedding-8B"
)
```

## 📊 Data Format Requirements

### Master File (Complete Data)
Must contain at least:
- Equipment identifier column (Equipment, equipment_number, etc.)
- Equipment Type or Equipment Description (or both)

### Target File (Incomplete Data) 
Must contain at least:
- Equipment identifier column
- May have missing Equipment Type and/or Equipment Description fields

### Supported Column Names
The system auto-detects these column variations:
- **Equipment ID**: Equipment, equipment_number, equip_no, equipment_id
- **Equipment Type**: Equipment Type, ObjectType, equipment_type, equip_type  
- **Equipment Description**: Equipment Description, equipment_description, description, desc

## ⚙️ Configuration

### GPU Settings
```python
# GPU configuration (automatic detection)
use_gpu = True              # Enable GPU acceleration
cuda_device = 0            # CUDA device index
max_memory_usage = 0.8     # Maximum GPU memory usage (80%)
batch_size = 16            # Optimized for RTX 3060 6GB
```

### Model Settings
```python
model_name = "Qwen/Qwen3-Embedding-8B"
max_seq_length = 512
torch_dtype = torch.float16  # FP16 for memory efficiency
```

### Search Settings
```python
similarity_threshold = 0.75  # Minimum similarity for matches
top_k_candidates = 5        # Number of candidates to consider
use_gpu_index = True        # GPU-accelerated FAISS index
index_type = "IVFFlat"      # Index algorithm
```

## 📈 Performance Optimization

### For RTX 3060 6GB
- Batch size: 16-32 (automatically configured)
- FP16 precision for memory efficiency
- Chunked processing for large datasets
- GPU memory monitoring and cleanup

### Memory Management
- Automatic batch size adjustment based on available VRAM
- Progressive cleanup of unused embeddings
- Efficient FAISS index memory usage

## 📝 Output Format

The completed dataset includes:
1. **Original Data**: All original columns preserved
2. **Filled Fields**: Missing Equipment Type and Description completed
3. **Similarity Scores**: Confidence scores for filled fields (optional)
4. **Completion Report**: JSON report with statistics

### Example Output Columns
```
equipment_number | equipment_type | equipment_description | equipment_type_similarity_score | equipment_description_similarity_score
```

## 🔍 Logging and Monitoring

Comprehensive logging includes:
- GPU memory usage tracking
- Processing time per stage
- Similarity matching statistics
- Model loading and cleanup status
- Error handling with detailed context

## 🧪 Example Use Case

Given ONGC equipment data:

**Master File (Complete)**:
```csv
equipment_number,equipment_type,equipment_description
000000000010000001,EQERMA,EQUALIZATION TANK MIXER MOTOR
000000000010000002,EQMRPC,100A01B-EQUALISATION TANK MIXERS
```

**Target File (Incomplete)**:
```csv
equipment_number,equipment_type,equipment_description
000000000010000003,,MIXER MOTOR FOR TANK
000000000010000004,EQMRPC,
```

**Completed Output**:
```csv
equipment_number,equipment_type,equipment_description,equipment_type_similarity_score
000000000010000003,EQERMA,MIXER MOTOR FOR TANK,0.87
000000000010000004,EQMRPC,100A01B-EQUALISATION TANK MIXERS,0.92
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 8`
   - Use CPU mode: `--cpu`
   - Close other GPU applications

2. **Model Loading Failed** 
   - Check internet connection for model download
   - Verify sufficient disk space (~16GB for model)
   - Ensure transformers library is updated

3. **FAISS Import Error**
   - Install both CPU and GPU versions: `pip install faiss-cpu faiss-gpu`
   - For CPU-only: `pip install faiss-cpu`

### Performance Tips

- Use GPU mode for datasets > 1000 records
- Increase batch size on higher VRAM GPUs
- Use IVFFlat index for datasets > 10K records
- Consider similarity threshold tuning (0.7-0.9)

## 🔄 Future Extensions

The modular design supports easy integration of:
- **LangChain**: For advanced prompting and chaining
- **ChromaDB**: Alternative vector database
- **Web Interface**: Streamlit/FastAPI integration
- **Batch Processing**: Multi-file processing capabilities
- **Fine-tuning**: Domain-specific model adaptation

## 📄 License

This project extends the ONGC Equipment Data Processing Pipeline with semantic completion capabilities.

## 🤝 Contributing

1. Maintain <150 lines per file limit
2. Add comprehensive logging
3. Include error handling
4. Update tests for new features
5. Document GPU/CPU compatibility
