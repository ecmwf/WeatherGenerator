"""
The entry point to train the weathergen model.
"""

# For profiling tools, the entry point cannot be in an __init__.py file.
from weathergen import train

if __name__ == "__main__":
    train()
