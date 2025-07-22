"""
Main entry point for semantic data completion pipeline
Command-line interface for the Qwen3-based equipment data completion system
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings, Settings
from pipeline import SemanticDataFiller
from utils import setup_logger


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Semantic Data Completion Pipeline using Qwen3 Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic usage with GPU acceleration
            python embedding.py master.xlsx target.xlsx

            # Specify output file and custom similarity threshold
            python embedding.py master.csv target.csv -o completed.xlsx -t 0.8

            # Force CPU usage and custom batch size
            python embedding.py master.xlsx target.xlsx --cpu --batch-size 8

            # Show device information
            python embedding.py --info
        """
    )

    parser.add_argument(
        "master_file",
        nargs="?",
        help="Path to master file with complete equipment data (CSV/Excel)"
    )

    parser.add_argument(
        "target_file",
        nargs="?",
        help="Path to target file with missing equipment data (CSV/Excel)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (optional, auto-generated if not specified)"
    )

    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for matching (0.0-1.0, default: 0.75)"
    )

    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        help="Batch size for processing (auto-detected if not specified)"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage (disable GPU acceleration)"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU usage (default if available)"
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-8B",
        help="HuggingFace model name (default: Qwen/Qwen3-Embedding-8B)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Don't include similarity scores in output"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show device and configuration information"
    )

    return parser


def show_device_info():
    """Display device and configuration information"""
    print("=" * 60)
    print("  SEMANTIC DATA COMPLETION PIPELINE")
    print("  Device and Configuration Information")
    print("=" * 60)

    # Get device info
    device_info = settings.get_device_info()

    print(f"GPU Acceleration: {'✅ Enabled' if device_info['use_gpu'] else '❌ Disabled'}")
    print(f"CUDA Available: {'✅ Yes' if device_info['cuda_available'] else '❌ No'}")
    print(f"Device: {device_info['device']}")
    print(f"Batch Size: {device_info['batch_size']}")

    if device_info['cuda_available']:
        print(f"CUDA Version: {device_info['cuda_version']}")
        print(f"GPU: {device_info['gpu_name']}")
        print(f"GPU Memory: {device_info['gpu_memory_gb']:.1f} GB")
        print(f"GPU Count: {device_info['gpu_count']}")

    print("\nModel Configuration:")
    print(f"Model: {settings.model.model_name}")
    print(f"Max Sequence Length: {settings.model.max_seq_length}")
    print(f"Model Device: {settings.model.device}")

    print("\nSearch Configuration:")
    print(f"Similarity Threshold: {settings.search.similarity_threshold}")
    print(f"Top K Candidates: {settings.search.top_k_candidates}")
    print(f"GPU Index: {'✅ Enabled' if settings.search.use_gpu_index else '❌ Disabled'}")
    print(f"Index Type: {settings.search.index_type}")

    print("\nSupported Formats:", ", ".join(settings.processing.supported_formats))
    print("=" * 60)


def validate_args(args) -> bool:
    """Validate command line arguments"""
    if args.info:
        return True

    if not args.master_file or not args.target_file:
        print("Error: Both master_file and target_file are required (unless using --info)")
        return False

    # Check if files exist
    master_path = Path(args.master_file)
    target_path = Path(args.target_file)

    if not master_path.exists():
        print(f"Error: Master file not found: {args.master_file}")
        return False

    if not target_path.exists():
        print(f"Error: Target file not found: {args.target_file}")
        return False

    # Check file formats
    supported_formats = settings.processing.supported_formats

    if master_path.suffix.lower() not in supported_formats:
        print(f"Error: Master file format not supported. Supported: {supported_formats}")
        return False

    if target_path.suffix.lower() not in supported_formats:
        print(f"Error: Target file format not supported. Supported: {supported_formats}")
        return False

    # Validate threshold
    if not (0.0 <= args.threshold <= 1.0):
        print("Error: Similarity threshold must be between 0.0 and 1.0")
        return False

    return True


def configure_settings(args):
    """Configure global settings based on arguments"""
    global settings

    # Determine GPU usage
    use_gpu = True
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True

    # Create new settings with custom parameters
    settings = Settings(
        use_gpu=use_gpu,
        similarity_threshold=args.threshold,
        batch_size=args.batch_size,
        custom_model_name=args.model
    )

    # Configure processing options
    if args.no_scores:
        settings.processing.include_similarity_scores = False


def main() -> int:
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Handle info request
    if args.info:
        show_device_info()
        return 0

    # Validate arguments
    if not validate_args(args):
        parser.print_help()
        return 1

    # Configure settings
    configure_settings(args)

    # Validate configuration
    if not settings.validate():
        print("Error: Invalid configuration")
        return 1

    # Setup logger
    logger = setup_logger(log_level=args.log_level)

    try:
        # Create and run pipeline
        with SemanticDataFiller(logger=logger) as pipeline:
            results = pipeline.process_files(
                master_file=args.master_file,
                target_file=args.target_file,
                output_file=args.output
            )

        # Handle results
        if results['status'] == 'completed':
            print("\n" + "=" * 60)
            print("  ✅ DATA COMPLETION SUCCESSFUL")
            print("=" * 60)
            print(f"Output File: {Path(results['output_file']).name}")
            print(f"Report File: {Path(results['report_file']).name}")

            stats = results['statistics']
            print(f"\nStatistics:")
            print(f"  Master Records: {stats['master_records']:,}")
            print(f"  Target Records: {stats['target_records']:,}")
            print(f"  Missing Fields: {stats['missing_fields_identified']}")

            index_stats = stats['index_stats']
            print(f"\nIndex Information:")
            print(f"  Total Vectors: {index_stats['total_vectors']:,}")
            print(f"  Dimensions: {index_stats['dimension']}")
            print(f"  Index Type: {index_stats['index_type']}")
            print(f"  GPU Accelerated: {'✅' if index_stats['is_gpu'] else '❌'}")

            print("=" * 60)
            return 0

        else:
            print("\n" + "=" * 60)
            print("  ❌ DATA COMPLETION FAILED")
            print("=" * 60)
            print(f"Error: {results['message']}")
            print("=" * 60)
            return 1

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\nUnexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
