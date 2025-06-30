#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install dependencies
pip install -r requirements.txt

# Create a directory for the NLTK data
mkdir -p nltk_data

# Download the NLTK data to the specified directory
python -m nltk.downloader -d nltk_data punkt
python -m nltk.downloader -d nltk_data stopwords