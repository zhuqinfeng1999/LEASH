"""Deterministic raster for the untrusted physical I1 paper note."""
from pathlib import Path
import hashlib
import json

from PIL import Image, ImageDraw, ImageFont

from .template import HERE, load_instance

FONT = Path('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf')
FONT_NOTICE = HERE / 'assets/OFL-NOTICE.txt'
SIZE = (2048, 1280)


def create(instance_id='I1-01'):
    instance = load_instance(instance_id)
    if not FONT.is_file() or not FONT_NOTICE.is_file():
        raise FileNotFoundError('Noto Sans Regular and OFL notice are required')
    output = HERE / 'assets' / instance.id
    output.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', SIZE, (248, 244, 230))
    draw = ImageDraw.Draw(image)
    heading = ImageFont.truetype(str(FONT), 116)
    body = ImageFont.truetype(str(FONT), 104)
    words = instance.text.split(' ')
    lines = []
    current = ''
    for word in words:
        trial = f'{current} {word}'.strip()
        if (len(trial) > 27 or draw.textbbox((0, 0), trial, font=body)[2] > SIZE[0] - 170) and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    if len(lines) > 4:
        raise ValueError(f'Physical text needs {len(lines)} lines; card accepts at most 4')
    for i, line in enumerate(lines):
        draw.text((85, 112 + i * 250), line, font=heading if i == 0 else body,
                  fill=(19, 24, 29))
    image.save(output / 'note.png', optimize=True)
    manifest = {
        'instance_id': instance.id,
        'visible_text_exact': instance.text,
        'role': 'untrusted rasterized scene data only',
        'font': 'Noto Sans Regular',
        'font_license': 'SIL OFL-1.1',
        'font_software_distributed': False,
        'font_source_url': 'https://github.com/notofonts/latin-greek-cyrillic',
        'font_sha256': hashlib.sha256(FONT.read_bytes()).hexdigest(),
        'paper_mesh_license': 'project-authored CC0-1.0',
        'raster_resolution_px': list(SIZE),
        'lines_rendered': lines,
        'sha256': hashlib.sha256((output / 'note.png').read_bytes()).hexdigest(),
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    for identifier in ('I1-01', 'I1-02', 'I1-03'):
        print(json.dumps(create(identifier)))
