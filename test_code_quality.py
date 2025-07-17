"""
Comprehensive test script to validate all modules and functionality
Tests modularity, imports, and basic functionality
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing module imports...")
    
    try:
        # Core modules
        import main
        print("✅ main.py imports successful")
        
        # Configuration
        from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
        print("✅ config.settings imports successful")
        
        from config.column_mappings import COLUMN_MAPPINGS
        print("✅ config.column_mappings imports successful")
        
        # Data processing
        from src.data.make_dataset import load_and_concatenate_data, clean_and_impute
        print("✅ src.data.make_dataset imports successful")
        
        # Features
        from src.features.build_features import feature_engineering
        print("✅ src.features.build_features imports successful")
        
        from src.features.feature_selection import optimize_feature_set
        print("✅ src.features.feature_selection imports successful")
        
        # Utils
        from src.utils.memory_utils import optimize_dtypes, PerformanceOptimizer
        print("✅ src.utils.memory_utils imports successful")
        
        from src.utils.logger import setup_logger
        print("✅ src.utils.logger imports successful")
        
        from src.utils.helpers import standardize_column_names
        print("✅ src.utils.helpers imports successful")
        
        # Dashboard
        import streamlit_dashboard
        print("✅ streamlit_dashboard imports successful")
        
        import run_dashboard
        print("✅ run_dashboard imports successful")
        
        print("🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_functionality():
    """Test core functionality"""
    print("\n🔍 Testing core functionality...")
    
    try:
        # Test sample data creation
        from main import create_sample_equipment_data
        df = create_sample_equipment_data()
        print(f"✅ Sample data created: {df.shape[0]} records, {df.shape[1]} columns")
        
        # Test memory optimization
        from src.utils.memory_utils import optimize_dtypes, get_memory_usage, PerformanceOptimizer
        optimized_df = optimize_dtypes(df.copy())
        print("✅ Memory optimization successful")
        
        # Test performance optimizer
        optimizer = PerformanceOptimizer()
        print("✅ PerformanceOptimizer initialized")
        
        # Test logger
        from src.utils.logger import setup_logger
        logger = setup_logger("test")
        logger.info("Test log message")
        print("✅ Logger working correctly")
        
        print("🎉 All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        traceback.print_exc()
        return False

def test_modularity():
    """Test code modularity and structure"""
    print("\n🔍 Testing modularity...")
    
    # Check for required directories
    required_dirs = [
        "src",
        "src/data", 
        "src/features",
        "src/utils",
        "config",
        "data",
        "data/Raw",
        "data/Processed"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories present")
    
    # Check for required files
    required_files = [
        "main.py",
        "requirements.txt", 
        "README.md",
        "streamlit_dashboard.py",
        "run_dashboard.py",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py", 
        "src/utils/__init__.py",
        "config/__init__.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
    
    print("🎉 Modularity check passed!")
    return True

def main():
    """Run all tests"""
    print("🚀 Starting comprehensive code validation...")
    print("=" * 60)
    
    # Run tests
    import_success = test_imports()
    modularity_success = test_modularity() 
    functionality_success = test_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"✅ Imports: {'PASS' if import_success else 'FAIL'}")
    print(f"✅ Modularity: {'PASS' if modularity_success else 'FAIL'}")
    print(f"✅ Functionality: {'PASS' if functionality_success else 'FAIL'}")
    
    if import_success and modularity_success and functionality_success:
        print("\n🎉 ALL TESTS PASSED! Code is modular and functional.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    main()
