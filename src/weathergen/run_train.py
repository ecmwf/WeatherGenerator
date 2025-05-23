"""
The entry point to train the weathergen model.
"""

# For profiling tools, the entry point cannot be in an __init__.py file.
# from weathergen import train
from weathergen.evaluation.test import test
from weathergen.utils.my_file import my_fun

def test_wrap():
    print(test())
    my_fun()

if __name__ == "__main__":
    print("Running test...")
    print(test())
    # train()
