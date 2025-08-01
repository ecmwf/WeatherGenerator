#import torch
import inspect
import json
import importlib
from types import ModuleType

OUTPUT_FILE = "torch_full_annotations.json"
DOMAIN_NAME = "PyTorch"

# Top-level PyTorch submodules to include
TOP_MODULES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch.cuda",
    "torch.autograd",
    "torch.optim",
    "torch.jit",
    "torch.distributions"
]

def extract_functions_from_module(module: ModuleType):
    functions = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            functions.append(name)
        elif inspect.isclass(obj):
            for method_name, method_obj in inspect.getmembers(obj):
                if inspect.isfunction(method_obj) or inspect.ismethod(method_obj):
                    functions.append(f"{obj.__name__}.{method_name}")
    return sorted(set(functions))

def safe_import(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"⚠️ Failed to import {module_name}: {e}")
        return None

annotations = []

for mod_name in TOP_MODULES:
    module = safe_import(mod_name)
    if module is None:
        continue
    functions = extract_functions_from_module(module)
    if functions:
        annotations.append({
            "domain": DOMAIN_NAME,
            "module": mod_name,
            "functions": functions
        })

# Save to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(annotations, f, indent=4)

print(f"✅ Generated annotations for {len(annotations)} PyTorch modules in {OUTPUT_FILE}")
