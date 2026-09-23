"""Run using a clean venv's Python -I; never imports from the source checkout."""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'src'
with tempfile.TemporaryDirectory(prefix='p2i-wheel-') as directory:
    os.chdir(directory)
    import p2i
    import torch
    from importlib.metadata import version
    from fastapi.testclient import TestClient
    from p2i.server.app import create_app
    assert not Path(p2i.__file__).resolve().is_relative_to(source)
    assert p2i.__version__ == version('p2i')
    subprocess.run([str(Path(sys.executable).parent / 'p2i'), '--help'], check=True)
    for demo in ('mlp', 'cnn', 'transformer'):
        subprocess.run([sys.executable, '-I', '-m', 'p2i.cli', 'demo', demo,
                        '--no-serve', '--save', f'{demo}.json'], check=True)
        ir = p2i.load(f'{demo}.json')
        assert ir.modules and ir.tensors
    client = TestClient(create_app(ir))
    page = client.get('/')
    assert page.status_code == 200
    assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', page.text)
    assert assets
    for asset in assets:
        response = client.get(asset)
        assert response.status_code == 200 and len(response.content) > 100
    assert client.get('/api/model').json()['version'] == ir.version
    print('PASS installed wheel: three offline CLI demos, model API and frontend assets')
