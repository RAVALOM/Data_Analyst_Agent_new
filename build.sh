#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install Python dependencies
pip install -r requirements.txt

# Run the model pre-caching script
python3 pre_cache_models.py
