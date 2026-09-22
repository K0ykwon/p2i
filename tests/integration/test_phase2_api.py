"""In-process ASGI transport also works in environments forbidding loopback sockets."""
import asyncio
import json
import subprocess
import sys
import httpx
import torch
from torch import nn
import p2i
from p2i.examples import make_demo
from p2i.server.app import create_app

def test_api_edit_roundtrip():
 model,args=make_demo('transformer');h=p2i.Harness(model,example_inputs=args);h.retrace(static=False)
 async def run():
  async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(h)),base_url='http://local') as client:
   before=(await client.get('/api/model')).json()
   assert (await client.get('/api/harness')).json()['enabled']
   action={'type':'set_parameter','target':'norm','parameter':'eps','value':.001,'expected_revision':0}
   preview=(await client.post('/api/harness/preview',json=action)).json()
   assert preview['result']['success'] and preview['state']['summary']['revision']==0
   committed=(await client.post('/api/harness/apply',json=action)).json()
   assert committed['state']['summary']['revision']==1
   assert committed['state']['summary']['observed_revision']==0
   assert (await client.get('/api/model')).json()==before
   stale=(await client.post('/api/harness/apply',json=action)).json()
   assert not stale['result']['success']
   updated=(await client.post('/api/harness/retrace',json={'expected_revision':1,'capture_values':True})).json()
   assert updated['result']['success'] and updated['state']['summary']['observed_revision']==1
   assert (await client.get('/api/inspection')).json()['tensors']
   report=(await client.post('/api/harness/validate',json={'backward':True})).json()
   assert report['result']['backward_valid']
   assert (await client.post('/api/harness/undo',json={'expected_revision':1})).json()['result']['success']
   assert (await client.post('/api/harness/redo',json={'expected_revision':2})).json()['result']['success']
   assert (await client.post('/api/harness/apply',json=action,headers={'Origin':'http://evil.example'})).status_code==403
 asyncio.run(run())

def test_observation_only_api():
 ir=p2i.trace(nn.Linear(2,1),(torch.ones(1,2),),static=False)
 async def run():
  async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(ir)),base_url='http://local') as client:
   assert (await client.get('/api/model')).json()['version']=='0.1'
   assert (await client.get('/api/harness')).json()=={'enabled':False}
   for kind in ('modules','operations','tensors'):
    assert (await client.get(f'/api/{kind}/{getattr(ir,kind)[0].id}')).status_code==200
   assert (await client.get('/api/modules/invalid')).status_code==404
   assert (await client.post('/api/harness/apply',json={})).status_code==409
 asyncio.run(run())

def test_cli_artifact_commands(tmp_path):
 h=p2i.Harness(nn.Sequential(nn.Linear(4,2),nn.Dropout(.1)))
 arch=tmp_path/'architecture.json';h.architecture().save(arch)
 action=tmp_path/'action.json';action.write_text(json.dumps({'type':'set_parameter','target':'1','parameter':'p','value':.2,'expected_revision':0}))
 out=tmp_path/'updated.json'
 def cli(*args):
  r=subprocess.run([sys.executable,'-m','p2i.cli','harness',*map(str,args)],capture_output=True,text=True,timeout=30)
  assert r.returncode==0,r.stderr+r.stdout
  return json.loads(r.stdout)
 assert cli('inspect',arch)['summary']['revision']==0
 assert cli('apply',arch,action,'--output',out)['success']
 assert cli('validate',out,'--input-shape',2,4)['forward_valid']
 assert cli('diff',arch,out)['changes'][0]['after']==.2
 assert not out.read_text()==arch.read_text()
