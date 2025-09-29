#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Setting up the environment..."

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

echo "Installing dependencies..."
# Install the required packages
pip install --upgrade pip
pip install -r requirements.txt

echo "Running the harmonization script..."
# Run the harmonization script
python harmonize.py --test_file "Test.xlsx" --output_file "Test_output.xlsx"

echo "Harmonization complete. Output saved to Test_output.xlsx"

# Deactivate the virtual environment
deactivate
