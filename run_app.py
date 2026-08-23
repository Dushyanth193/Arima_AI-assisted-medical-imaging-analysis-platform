"""
run_app.py
----------
Root launcher script for the Nexora OrthoAI Platform (Module 1).
Starts the local Streamlit web application.
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_dir, "module1_mri", "app.py")

    venv_python = os.path.join(base_dir, "module1_mri", ".venv", "Scripts", "python.exe")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    cmd = [python_exe, "-m", "streamlit", "run", app_path, "--server.port=8501", "--server.headless=true"]
    print(f"Launching Nexora OrthoAI Web Application at http://localhost:8501 ...")
    print(f"Using Python executable: {python_exe}")
    print(f"App Script: {app_path}")
    subprocess.run(cmd)
