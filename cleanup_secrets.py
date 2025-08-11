import glob
import json
import re
import os

def clean_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    changed = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell['source'])
        new_source = source

        # Replace explicit os.environ assignments
        new_source = re.sub(
            r"os\\.environ\\[['\"]([^'\"]+)['\"]\\]\\s*=\\s*['\"][^'\"]+['\"]",
            lambda m: f"# {m.group(0)}\nimport os\n{m.group(1)} = os.getenv(\"{m.group(1)}\")",
            new_source
        )

        # Replace direct API key assignments (e.g., OPENAI_API_KEY = "sk-...")
        new_source = re.sub(
            r"([A-Za-z0-9_]+)\\s*=\\s*['\"][A-Za-z0-9_/+=]+['\"]",
            lambda m: f"# {m.group(0)}\n{m.group(1)} = os.getenv(\"{m.group(1)}\")",
            new_source
        )

        if new_source != source:
            cell['source'] = new_source.splitlines(keepends=True)
            changed = True

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print(f"Cleaned secrets in {path}")

if __name__ == "__main__":
    # Walk through notebooks in showcase, experiments, and archive
    for root_dir in ['showcase', 'experiments', 'archive']:
        for path in glob.glob(os.path.join(root_dir, '**', '*.ipynb'), recursive=True):
            clean_notebook(path)
