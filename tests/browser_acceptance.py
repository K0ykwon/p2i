"""Launch real servers and run legacy + Phase 2 Playwright acceptance.

Explicit script, outside pytest discovery. Requires sockets and installed Chromium.
"""
from pathlib import Path
import os
import subprocess
import sys
import time
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
def run(name,edit,script):
 command=[sys.executable,'-m','p2i.cli','demo',name,'--port','8000']
 if edit:command+=['--edit']
 registry_dir=tempfile.TemporaryDirectory(prefix='p2i-browser-registry-')
 env={**os.environ,'P2I_SKILL_REGISTRY':str(Path(registry_dir.name)/'skills.sqlite3'),'P2I_DEMO':name}
 server=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,env=env)
 try:
  for _ in range(600):
   if server.poll() is not None:raise RuntimeError(server.stderr.read().decode())
   try:urllib.request.urlopen('http://127.0.0.1:8000/api/model',timeout=1);break
   except OSError:time.sleep(.1)
  else:raise RuntimeError('Local server did not become ready')
  subprocess.run(['node',f'tests/{script}'],cwd=ROOT/'frontend',env=env,check=True,timeout=90)
 finally:
  server.terminate()
  try:server.wait(timeout=5)
  except subprocess.TimeoutExpired:server.kill();server.wait()
  registry_dir.cleanup()
if __name__=='__main__':
 run('transformer',False,'explorer.mjs')
 for name in ('transformer','cnn','mlp'):run(name,True,'phase2.mjs')
 run('transformer',True,'phase3.mjs')
 for name in ('transformer','cnn','mlp'):run(name,True,'ux.mjs')
