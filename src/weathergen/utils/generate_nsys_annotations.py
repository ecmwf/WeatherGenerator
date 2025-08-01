import os
import ast
import json
from pathlib import Path

# Change this to the path of your code base
SOURCE_ROOT = Path("/p/project1/hclimrep/kasravi1/WeatherGenerator/src/weathergen")
OUTPUT_FILE = Path("/p/project1/hclimrep/kasravi1/WeatherGenerator/out.json")
DOMAIN_NAME = "WeatherGen"

def get_module_path(file_path: Path, root_path: Path):
    rel_path = file_path.relative_to(root_path.parent)
    return ".".join(rel_path.with_suffix('').parts)

def extract_functions_from_file(file_path):
    functions = []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError:
            print(f"Skipping invalid syntax: {file_path}")
            return []
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    functions.append(f"{class_name}.{item.name}")
    return functions

def generate_annotations_json(source_root, domain):
    annotations = []
    for file_path in source_root.rglob("*.py"):
        if file_path.name == "__init__.py":
            continue
        module = get_module_path(file_path, source_root)
        functions = extract_functions_from_file(file_path)
        if functions:
            annotations.append({
                "domain": domain,
                "module": module,
                "functions": functions
            })
    return annotations

# Generate and write annotations
annotations = generate_annotations_json(SOURCE_ROOT, DOMAIN_NAME)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(annotations, f, indent=4)

print(f"✅ Generated {len(annotations)} modules in {OUTPUT_FILE}")
