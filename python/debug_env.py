#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

print("== Python Environment Debug Information ==")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

# Check for required libraries
print("\n== Library Availability ==")
libraries = ["librosa", "numpy", "sklearn", "joblib"]
for lib in libraries:
    try:
        if lib == "sklearn":
            import sklearn
            print(f"✅ {lib} imported successfully (version: {sklearn.__version__})")
        elif lib == "librosa":
            import librosa
            print(f"✅ {lib} imported successfully (version: {librosa.__version__})")
        elif lib == "numpy":
            import numpy
            print(f"✅ {lib} imported successfully (version: {numpy.__version__})")
        elif lib == "joblib":
            import joblib
            print(f"✅ {lib} imported successfully (version: {joblib.__version__})")
    except ImportError:
        print(f"❌ {lib} import failed")
    except Exception as e:
        print(f"⚠️ {lib} error: {str(e)}")

# Check model files
print("\n== Model File Availability ==")
model_files = [
    os.path.join("python", "advanced_model.pkl"),
    os.path.join("python", "advanced_scaler.pkl"),
    os.path.join("python", "advanced_nn_model.pt")
]

for model_file in model_files:
    if os.path.exists(model_file):
        print(f"✅ {model_file} exists ({os.path.getsize(model_file)} bytes)")
    else:
        print(f"❌ {model_file} does not exist")

print("\nEnd of debug information")
