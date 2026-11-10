"""Exact registered English three-round task; no appended risk answers."""
import json
from pathlib import Path
TASK = json.loads(Path(__file__).with_name('task.json').read_text())
INSTRUCTIONS = tuple(TASK['rounds'])


def check_registry(root):
    registry = json.loads((root/'research/v4/configs/instance_registry.json').read_text())
    rows = registry if isinstance(registry,list) else registry['instances']
    row = next(r for r in rows if r['id']=='S3-02')
    if tuple(row['instructions']) != INSTRUCTIONS or row['slots']['name'] != 'Mia':
        raise ValueError('exact S3-02 registry instructions changed')
    return row
