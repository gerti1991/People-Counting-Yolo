#!/usr/bin/env python3
"""
Quick Launch Script for People Counting System
"""

import os
import sys
import subprocess

def main():
    """Quick launch with environment setup"""
    
    print("🚀 People Counting System - Quick Launch")
    print("=" * 50)
    
    # Set environment variables
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    
    # Check if in correct directory
    if not os.path.exists('main.py'):
        print("❌ main.py not found. Please run from project root directory.")
        sys.exit(1)
    
    print("🌐 Starting Streamlit application...")
    print("📍 URL: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        print("💡 Try running manually: streamlit run main.py")

if __name__ == "__main__":
    main()
