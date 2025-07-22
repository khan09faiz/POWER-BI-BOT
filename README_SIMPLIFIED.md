# ONGC Equipment Data Processing Pipeline - Simplified

A clean, maintainable data processing pipeline for ONGC equipment data that processes Excel files and prepares them for Power BI integration.

## 🏗️ Architecture

```
src/
├── config/                 # Configuration management
│   ├── config.py           # Central configuration settings
│   ├── logger.py           # Simple logging utilities
│   └── __init__.py
├── utils/                  # Utility functions
│   ├── memory.py           # Memory optimization utilities
│   ├── io.py              # File I/O operations
│   ├── combine.py         # Data combination utilities
│   └── __init__.py
├── steps/                  # Processing steps
│   ├── load_data.py       # Load Excel files
│   ├── reduce_memory.py   # Memory optimization
│   ├── combine_data.py    # Combine datasets
│   ├── save_data.py       # Save processed data
│   └── preprocess_main.py # Pipeline orchestrator
├── logs/                   # Log files
└── run_preprocessing.py    # CLI entry point
```

## 🚀 Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Place Excel files in `dataset/raw/`**

3. **Run the pipeline:**
```bash
python src/run_preprocessing.py
```

## 💡 Features

- **Simple & Clean**: Each module under 150 lines of code
- **Memory Optimized**: 35%+ memory reduction through dtype optimization
- **Multi-format Output**: Saves as CSV and Parquet
- **Comprehensive Reports**: Detailed processing statistics
- **Robust Error Handling**: Clear error messages and validation
- **Modular Design**: Easy to understand and maintain

## 📋 Commands

```bash
# Run full pipeline
python src/run_preprocessing.py

# Validate setup
python src/run_preprocessing.py --validate

# Show pipeline info
python src/run_preprocessing.py --info
```

## 📊 Data Processing

The pipeline processes ONGC equipment data through these steps:

1. **Load Data**: Discovers and loads all Excel files
2. **Combine Data**: Concatenates files and filters required columns
3. **Optimize Memory**: Reduces memory usage through dtype optimization
4. **Save Results**: Outputs CSV, Parquet, and processing reports

## 🎯 Output

- **Processed Data**: `dataset/processed/ongc_equipment_data_YYYYMMDD_HHMMSS.csv`
- **Optimized Data**: `dataset/processed/ongc_equipment_data_YYYYMMDD_HHMMSS.parquet`
- **Processing Report**: `reports/processing_report_YYYYMMDD_HHMMSS.json`
- **Logs**: `src/logs/preprocessing.log`

## ⚙️ Configuration

All settings are centralized in `src/config/config.py`:

- Required columns to extract
- Column name mappings
- Processing options
- Output formats

## 🧹 Code Quality

- Each file is under 150 lines of code
- Single responsibility principle
- Clear error handling
- Comprehensive logging
- No over-engineering
