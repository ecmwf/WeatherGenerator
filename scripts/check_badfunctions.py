# ruff: noqa: T201, N802

"""
Check for usage of banned functions in Python source files.

This script uses the Abstract Syntax Tree (AST) module to parse Python files
and detect calls to functions that are explicitly banned in the codebase.
"""

import ast
import sys
from pathlib import Path

# List of function names that are not allowed in the codebase
BANNED_FUNCTIONS: set[str] = {
    "getattr",
    # "setattr",
    # "delattr",
    # "eval",
    # "exec",
    # Add more banned functions here as needed
}


class BannedFunctionVisitor(ast.NodeVisitor):
    """
    AST visitor that detects calls to banned functions.

    This visitor walks through the Abstract Syntax Tree of a Python file
    and records any calls to functions that are in the BANNED_FUNCTIONS set.
    """

    def __init__(self, filepath: Path):
        """
        Initialize the visitor.

        Args:
            filepath: Path to the file being checked (for error reporting)
        """
        self.filepath = filepath
        self.errors: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        """
        Visit a function call node in the AST.

        This method is called automatically by the AST walker whenever
        it encounters a function call. We check if the called function
        is in our banned list.

        Args:
            node: The AST Call node representing a function call
        """
        # Check if this is a simple function call (e.g., getattr(...))
        # as opposed to a method call (e.g., obj.method(...))
        if isinstance(node.func, ast.Name):
            function_name = node.func.id

            # If the function is banned, record an error
            if function_name in BANNED_FUNCTIONS:
                error_msg = (
                    f"{self.filepath}:{node.lineno}:{node.col_offset}: "
                    f"Use of banned function '{function_name}()' is not allowed"
                )
                self.errors.append(error_msg)

        # Continue visiting child nodes
        self.generic_visit(node)


def check_file(filepath: Path) -> list[str]:
    """
    Check a single Python file for banned function usage.

    Args:
        filepath: Path to the Python file to check

    Returns:
        List of error messages (empty if no banned functions found)
    """
    try:
        # Read and parse the Python file into an AST
        with open(filepath, encoding="utf-8") as f:
            source_code = f.read()

        tree = ast.parse(source_code, filename=str(filepath))

    except SyntaxError as e:
        # If the file has syntax errors, report them but don't fail
        # (syntax errors will be caught by other linters)
        return [f"{filepath}: Syntax error, skipping banned function check: {e}"]

    except Exception as e:
        # Handle other unexpected errors (e.g., encoding issues)
        return [f"{filepath}: Error reading file: {e}"]

    # Create a visitor and walk through the AST
    visitor = BannedFunctionVisitor(filepath)
    visitor.visit(tree)

    return visitor.errors


def main() -> int:
    """
    Main entry point for the banned functions checker.

    Scans all Python files in the configured directories and reports
    any usage of banned functions.

    Returns:
        Exit code: 0 if no banned functions found, 1 otherwise
    """
    # Directories to check (relative to the script's parent directory)
    script_dir = Path(__file__).parent.parent
    directories_to_check = ["src/", "scripts/", "packages/"]

    all_errors: list[str] = []

    # Walk through each directory and check all Python files
    for directory in directories_to_check:
        dir_path = script_dir / directory

        if not dir_path.exists():
            print(f"Warning: Directory {directory} does not exist, skipping")
            continue

        # Find all .py files recursively
        for py_file in dir_path.rglob("*.py"):
            errors = check_file(py_file)
            all_errors.extend(errors)

    # Print all errors found
    if all_errors:
        print("=" * 70)
        print("BANNED FUNCTION USAGE DETECTED")
        print("=" * 70)
        for error in all_errors:
            print(error)
        print("=" * 70)
        print(f"\nTotal violations: {len(all_errors)}")
        print(f"Banned functions: {', '.join(sorted(BANNED_FUNCTIONS))}")
        return 1

    print("✓ No banned functions detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
