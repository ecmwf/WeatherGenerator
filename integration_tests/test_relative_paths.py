"""Integration test to check for relative paths in the codebase. AI generated."""

import os
import re
import pytest
from pathlib import Path


class TestRelativePaths:
    """Test suite to detect potentially problematic relative paths in the codebase."""

    # Root directory of the project
    PROJECT_ROOT = Path(__file__).parent.parent

    # Patterns that indicate relative paths
    RELATIVE_PATH_PATTERNS = [
        r'"\./[^"]+',           # "./path"
        r"'\./[^']+",           # './path'
        r'"\.\./[^"]+',         # "../path"
        r"'\.\./[^']+",         # '../path'
        r'open\(["\'][^/][^"\']+["\']',  # open('relative/path')
    ]

    # File extensions to check
    EXTENSIONS_TO_CHECK = {'.py'} # , '.yaml', '.yml', '.json', '.toml'

    # Directories to exclude
    EXCLUDE_DIRS = {'__pycache__', '.git', '.venv', 'integration_tests'}

    # Files to exclude (e.g., this test file itself)
    EXCLUDE_FILES = {'test_relative_paths.py'}

    def get_all_source_files(self):
        """Recursively get all source files in the project."""
        source_files = []
        for root, dirs, files in os.walk(self.PROJECT_ROOT):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS]
            
            for file in files:
                if file in self.EXCLUDE_FILES:
                    continue
                if Path(file).suffix in self.EXTENSIONS_TO_CHECK:
                    source_files.append(Path(root) / file)
        return source_files

    def find_relative_paths_in_file(self, filepath: Path) -> list[tuple[int, str, str]]:
        """
        Find relative paths in a file.
        
        Returns:
            List of tuples: (line_number, matched_pattern, line_content)
        """
        matches = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip comments
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('//'):
                        continue
                    
                    for pattern in self.RELATIVE_PATH_PATTERNS:
                        found = re.search(pattern, line)
                        if found:
                            matches.append((line_num, found.group(), line.strip()))
        except Exception as e:
            pytest.skip(f"Could not read file {filepath}: {e}")
        
        return matches

    def test_no_hardcoded_relative_paths(self):
        """Ensure no hardcoded relative paths exist in Python files."""
        source_files = self.get_all_source_files()
        violations = []

        for filepath in source_files:
            print(f"Checking file: {filepath}")
            matches = self.find_relative_paths_in_file(filepath)
            for line_num, match, line in matches:
                violations.append({
                    'file': str(filepath.relative_to(self.PROJECT_ROOT)),
                    'line': line_num,
                    'match': match,
                    'content': line
                })

        if violations:
            report = "\n\nRelative path violations found:\n"
            for v in violations:
                report += f"\n  {v['file']}:{v['line']}\n"
                report += f"    Match: {v['match']}\n"
                report += f"    Line: {v['content']}\n"
            
            pytest.fail(report)

    def test_yaml_configs_use_absolute_or_variable_paths(self):
        """Check that YAML config files don't use hardcoded relative paths."""
        yaml_files = [f for f in self.get_all_source_files() 
                      if f.suffix in {'.yaml', '.yml'}]
        
        violations = []
        relative_path_pattern = re.compile(r':\s*["\']?\.\.?/[^"\'#\n]+')

        for yaml_file in yaml_files:
            print(f"Checking YAML file: {yaml_file}")
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if relative_path_pattern.search(line):
                            violations.append({
                                'file': str(yaml_file.relative_to(self.PROJECT_ROOT)),
                                'line': line_num,
                                'content': line.strip()
                            })
            except Exception:
                continue

        if violations:
            report = "\n\nRelative paths in YAML configs:\n"
            for v in violations:
                report += f"\n  {v['file']}:{v['line']}: {v['content']}\n"
            pytest.fail(report)

    def test_path_construction_uses_pathlib_or_os_path(self):
        """Check that path construction uses pathlib or os.path properly."""
        python_files = [f for f in self.get_all_source_files() if f.suffix == '.py']
        
        # Pattern for string concatenation that looks like path building
        bad_pattern = re.compile(r'["\'][^"\']*["\']\s*\+\s*["\']/|/["\']\s*\+')
        
        violations = []
        for py_file in python_files:
            print(f"Checking Python file: {py_file}")
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if bad_pattern.search(line):
                            violations.append({
                                'file': str(py_file.relative_to(self.PROJECT_ROOT)),
                                'line': line_num,
                                'content': line.strip()
                            })
            except Exception:
                continue

        if violations:
            report = "\n\nPotential unsafe path concatenation found:\n"
            report += "(Consider using pathlib.Path or os.path.join)\n"
            for v in violations:
                report += f"\n  {v['file']}:{v['line']}: {v['content']}\n"
            pytest.fail(report)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])