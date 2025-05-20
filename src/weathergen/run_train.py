"""
The entry point to train the weathergen model.
"""

# For profiling tools, the entry point cannot be in an __init__.py file.
from weathergen import train
import os

if __name__ == "__main__":
    print("Starting training...")
    # PRint the environment variables
    print("Environment variables:")
    for key, value in os.environ.items():
        print(f"{key}: {value}")
    train()
