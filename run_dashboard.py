"""
Quick launcher for the ONGC Equipment Analysis Dashboard
Run this to start the Streamlit dashboard
"""

import subprocess
import sys
from pathlib import Path

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    
    print("🚀 Launching ONGC Equipment Analysis Dashboard...")
    print("=" * 50)
    
    # Check if we're in the right directory
    dashboard_path = Path("streamlit_dashboard.py")
    if not dashboard_path.exists():
        print("❌ streamlit_dashboard.py not found!")
        print("   Make sure you're running this from the POWER-BI-BOT directory")
        return
    
    # Check for data files
    from config.settings import PROCESSED_DATA_DIR, FEATURES_DATA_DIR
    
    data_found = False
    if PROCESSED_DATA_DIR.exists():
        data_files = list(PROCESSED_DATA_DIR.glob("*.parquet")) + list(PROCESSED_DATA_DIR.glob("*.csv"))
        if data_files:
            data_found = True
            print(f"✅ Found {len(data_files)} processed data files")
    
    if not data_found:
        print("⚠️  No processed data found!")
        print("   Consider running: python main.py")
        print("   to generate cleaned data first")
        print()
    
    # Launch Streamlit
    try:
        print("🌐 Starting Streamlit server...")
        print("   Dashboard will open in your default browser")
        print("   Press Ctrl+C to stop the server")
        print()
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py",
            "--theme.base", "light"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\n💡 Try running manually:")
        print("   streamlit run streamlit_dashboard.py")

if __name__ == "__main__":
    launch_dashboard()
