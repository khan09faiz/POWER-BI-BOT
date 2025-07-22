"""
Main Pipeline orchestrator for ONGC Equipment Data Processing
Enhanced with modular preprocessing architecture
"""

import sys
import traceback
import time
from typing import Dict, Any, Optional
import logging
from enum import Enum
from .config import get_config, load_config
from .logger import setup_logging, log_system_info, get_performance_logger
from .data_preprocessor import create_preprocessor
from .data_saver import create_data_saver

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Enhanced circuit breaker implementation for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, name: str = "default"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.name = name
        self.success_count = 0
        self.total_calls = 0
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        self.total_calls += 1
        
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logging.getLogger('ongc_pipeline').info(f"🔄 Circuit breaker '{self.name}' moved to HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.success_count += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Reset after successful call in half-open state
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
            logging.getLogger('ongc_pipeline').info(f"✅ Circuit breaker '{self.name}' reset to CLOSED state")
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on successful calls
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logging.getLogger('ongc_pipeline').warning(f"⚠️ Circuit breaker '{self.name}' opened due to {self.failure_count} failures")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "success_rate": self.success_count / self.total_calls if self.total_calls > 0 else 0,
            "last_failure_time": self.last_failure_time
        }

class ONGCDataPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(
            log_level=self.config.log_level,
            logs_dir=self.config.logs_dir
        )
        
        # Initialize components
        self.preprocessor = create_preprocessor(self.config)
        self.data_saver = create_data_saver(self.config)
        self.perf_logger = get_performance_logger()
        
        # Initialize circuit breakers if enabled
        if getattr(self.config, 'enable_circuit_breaker', False):
            self.preprocessor_cb = CircuitBreaker(
                failure_threshold=getattr(self.config, 'circuit_breaker_failure_threshold', 5),
                recovery_timeout=getattr(self.config, 'circuit_breaker_recovery_timeout', 60),
                name="preprocessor"
            )
            self.data_saver_cb = CircuitBreaker(
                failure_threshold=getattr(self.config, 'circuit_breaker_failure_threshold', 5),
                recovery_timeout=getattr(self.config, 'circuit_breaker_recovery_timeout', 60),
                name="data_saver"
            )
        else:
            self.preprocessor_cb = None
            self.data_saver_cb = None
        
        # Log system information
        log_system_info(self.logger)
    
    def run(self) -> Dict[str, Any]:
        """Run the complete data processing pipeline with retry logic"""
        max_retries = getattr(self.config, 'max_retry_attempts', 3)
        
        for attempt in range(max_retries):
            try:
                self.logger.info("=" * 80)
                self.logger.info("🚀 ONGC EQUIPMENT DATA PROCESSING PIPELINE STARTED")
                if attempt > 0:
                    self.logger.info(f"   Retry attempt {attempt + 1}/{max_retries}")
                self.logger.info("=" * 80)
                
                return self._execute_pipeline()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"⚠️ Pipeline attempt {attempt + 1} failed: {str(e)}")
                    self.logger.info(f"🔄 Retrying... ({attempt + 2}/{max_retries})")
                    continue
                else:
                    self.logger.error("❌ All retry attempts exhausted")
                    raise
    
    def _execute_pipeline(self) -> Dict[str, Any]:
        """Execute the main pipeline logic with new preprocessor"""
        try:
            
            # Step 1: Preprocess data (load, concatenate, filter, clean) with circuit breaker protection
            self.logger.info("� Step 1: Data Preprocessing (Load → Filter → Clean)")
            if self.preprocessor_cb:
                processed_df, preprocessing_stats = self.preprocessor_cb.call(self.preprocessor.preprocess_data)
            else:
                processed_df, preprocessing_stats = self.preprocessor.preprocess_data()
            
            if processed_df is None or processed_df.empty:
                self.logger.error("❌ Data preprocessing failed or resulted in empty dataset. Pipeline terminated.")
                return self._create_failure_result("Data preprocessing failed")
            
            self.logger.info(f"✅ Data preprocessing completed: {processed_df.shape}")
            
            # Step 2: Save processed data with circuit breaker protection
            self.logger.info("� Step 2: Saving processed data")
            if self.data_saver_cb:
                saved_files = self.data_saver_cb.call(self.data_saver.save_processed_data, processed_df, preprocessing_stats.__dict__)
            else:
                saved_files = self.data_saver.save_processed_data(processed_df, preprocessing_stats.__dict__)
            
            # Step 3: Generate comprehensive report
            self.logger.info("� Step 3: Generating processing report")
            performance_metrics = self.perf_logger.get_summary()
            
            # Create final result
            result = self._create_success_result(processed_df, preprocessing_stats, saved_files, performance_metrics)
            
            self.logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._create_failure_result(str(e))
            
            processing_report = self.data_saver.create_processing_report(
                files_processed=files_processed,
                processing_summary=data_summary,
                performance_metrics=performance_metrics,
                saved_files=saved_files
            )
            
            report_path = self.data_saver.save_processing_report(processing_report)
            
            # Final success summary
            self.logger.info("=" * 80)
            self.logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self.logger.info(f"📊 Records processed: {len(processed_df):,}")
            self.logger.info(f"📋 Columns: {len(processed_df.columns)}")
            self.logger.info(f"📁 Files saved: {len(saved_files)}")
            self.logger.info(f"⏱️ Total execution time: {performance_metrics['total_execution_time']:.2f}s")
            self.logger.info(f"🧠 Memory usage: {performance_metrics['current_memory_mb']:.1f}MB")
            self.logger.info("=" * 80)
            
            # Print file locations for user
            self.logger.info("📁 Output files:")
            for file_type, file_path in saved_files.items():
                self.logger.info(f"   {file_type}: {file_path}")
            
            if report_path:
                self.logger.info(f"   processing_report: {report_path}")
            
            return self._create_success_result(
                processed_df=processed_df,
                data_summary=data_summary,
                saved_files=saved_files,
                performance_metrics=performance_metrics,
                report_path=report_path
            )
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error("❌ PIPELINE FAILED!")
            self.logger.error("=" * 80)
            self.logger.error(f"Error: {str(e)}")
            self.logger.error("Traceback:")
            self.logger.error(traceback.format_exc())
            self.logger.error("=" * 80)
            
            return self._create_failure_result(str(e))
    
    def _create_success_result(self, 
                             processed_df,
                             preprocessing_stats,
                             saved_files: Dict[str, str],
                             performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create success result dictionary"""
        return {
            "status": "success",
            "message": "Pipeline completed successfully",
            "data_shape": processed_df.shape,
            "records_processed": len(processed_df),
            "columns_count": len(processed_df.columns),
            "preprocessing_stats": preprocessing_stats.__dict__ if hasattr(preprocessing_stats, '__dict__') else preprocessing_stats,
            "saved_files": saved_files,
            "performance_metrics": performance_metrics,
            "execution_time": performance_metrics.get('total_execution_time', 0)
        }
    
    def _create_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create failure result dictionary"""
        performance_metrics = self.perf_logger.get_summary()
        
        return {
            "status": "failed",
            "message": f"Pipeline failed: {error_message}",
            "error": error_message,
            "performance_metrics": performance_metrics,
            "execution_time": performance_metrics.get('total_execution_time', 0)
        }
    
    def get_config(self):
        """Get current configuration"""
        return self.config
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate pipeline setup and requirements"""
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check if raw data directory exists
        from pathlib import Path
        raw_dir = Path(self.config.raw_data_dir)
        if not raw_dir.exists():
            validation_results["warnings"].append(f"Raw data directory does not exist: {raw_dir}")
        
        # Check for data files
        files = self.data_loader.discover_files()
        if not files:
            validation_results["issues"].append("No data files found in raw data directory")
            validation_results["valid"] = False
        
        # Check write permissions
        try:
            test_file = Path(self.config.processed_data_dir) / "test_write.tmp"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            validation_results["issues"].append(f"Cannot write to output directory: {e}")
            validation_results["valid"] = False
        
        return validation_results

def create_pipeline(config_path: str = None) -> ONGCDataPipeline:
    """Factory function to create pipeline instance"""
    return ONGCDataPipeline(config_path)

def run_pipeline(config_path: str = None) -> Dict[str, Any]:
    """Convenience function to create and run pipeline"""
    pipeline = create_pipeline(config_path)
    return pipeline.run()