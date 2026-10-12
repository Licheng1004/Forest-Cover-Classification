"""Check notebook JSON and compile Python code cells without executing them."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
notebooks = sorted(ROOT.rglob('*.ipynb'))
if not notebooks:
    raise SystemExit('No Jupyter notebooks found.')

for path in notebooks:
    notebook = json.loads(path.read_text(encoding='utf-8'))
    if notebook.get('nbformat') != 4 or not isinstance(notebook.get('cells'), list):
        raise SystemExit(f'Invalid or unsupported notebook: {path.relative_to(ROOT)}')
    for index, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        source = ''.join(source) if isinstance(source, list) else source
        lines = source.splitlines(keepends=True)
        first_code = next((line.lstrip() for line in lines if line.strip()), '')
        if first_code.startswith('%%'):
            continue
        python_lines = [line for line in lines if not line.lstrip().startswith(('%', '!'))]
        try:
            ast.parse(''.join(python_lines), filename=f'{path.name}:cell {index}')
        except SyntaxError as error:
            raise SystemExit(
                f'{path.relative_to(ROOT)} cell {index}: '
                f'{error.msg} (line {error.lineno})'
            ) from error

print(f'Validated JSON and Python syntax in {len(notebooks)} notebook(s).')
