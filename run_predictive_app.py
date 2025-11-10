#!/usr/bin/env python3
"""
CamDoc Predictive Analysis Launcher

This script launches the Streamlit application for predictive features analysis.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    
    # Get the app path
    app_path = Path(__file__).parent / "apps" / "predictive_analysis.py"
    
    if not app_path.exists():
        print(f"❌ Error: App file not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting CamDoc Predictive Analysis App...")
    print(f"📂 App location: {app_path}")
    print("🌐 The app will open in your default web browser")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
