import json
import os
import subprocess
import sys
import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient
import p2i
from p2i.examples import make_demo
from p2i.examples.skill_discovery import run,candidate
from p2i.skills import SkillRegistry,SkillError
from p2i.server.app import create_app

@pytest.mark.parametrize('name,target',[('mlp','layers.1'),('cnn','features.1'),('transformer','blocks.0.mlp.1')])
def test_demo_skill_acceptance(tmp_path,name,target):
 model,inputs=make_demo(name)
 h=p2i.Harness(model,example_inputs=inputs,skill_registry=SkillRegistry(tmp_path/'r.db'))
 h.retrace(static=False)
 action=p2i.ReplaceWithSkill(target=target,skill_id='builtin.silu')
 assert h.preview(action).success
 assert h.apply(action).success
 assert h.validate(backward=True).backward_valid
 assert h.retrace(static=False).modules
 assert h.architecture().revision==h.observed_revision==1
 assert h.evaluate().metrics['forward_success']==1

def test_discovery_repair_and_reuse(tmp_path):
 summary=run(tmp_path)
 assert summary['success'] and summary['model_b_revision']==1
 registry=SkillRegistry(tmp_path/'skills.sqlite3')
 promoted=registry.get(summary['skill_id'])
 assert promoted.status=='promoted' and promoted.validation.valid and len(promoted.evaluation)==1
 assert len(registry.versions(promoted.id))>=6
 h=p2i.Harness(nn.Sequential(nn.Identity()),example_inputs=(torch.randn(2,7,8),),skill_registry=registry)
 failed=h.apply(p2i.ReplaceWithSkill(target='0',skill_id=promoted.id))
 assert not failed.success and failed.error_code=='LOCAL_CODE_OPT_IN_REQUIRED'
 assert registry.get(promoted.id,0).status=='candidate'
 changed=candidate();changed.implementation.source+='\n# Revised source requires validation\n'
 new=registry.update(changed)
 assert new.status=='candidate' and new.validation is None
 with pytest.raises(SkillError):registry.promote(new.id)

def test_api_skill_views_and_tool_errors(tmp_path):
 model,args=make_demo('transformer');registry=SkillRegistry(tmp_path/'r.db')
 registry.register(candidate())
 h=p2i.Harness(model,example_inputs=args,skill_registry=registry);h.retrace(static=False)
 with TestClient(create_app(h)) as client:
  assert len(client.get('/api/skills').json()['skills'])>=20
  assert client.get('/api/skills/discovered.gated_residual').json()['skill']['status']=='candidate'
  node=h.observe(node_id='blocks.0.mlp.1')
  options=client.get('/api/skills/compatible/'+node['id']).json()['skills']
  assert any(s['skill']['id']=='builtin.silu' for s in options)
  a={'type':'replace_with_skill','target':node['id'],'skill_id':'builtin.silu'}
  for cmd in ('preview','apply'):
   response=client.post('/api/harness/'+cmd,json=a)
   assert response.json()['result']['success']
  assert client.post('/api/harness/retrace',json={}).json()['result']['success']
  assert client.post('/api/tools',json={'tool':'inspect_model'}).json()['success']
  assert client.post('/api/tools',json={'tool':'nonexistent'}).json()['error']['code']=='UNKNOWN_TOOL'
  assert client.post('/api/tools',json={'tool':'inspect_model'},headers={'origin':'https://other.invalid'}).status_code==403

def test_cli_acceptance(tmp_path):
 env={**os.environ,'P2I_SKILL_REGISTRY':str(tmp_path/'registry.db')}
 def cli(*args,success=True):
  p=subprocess.run([sys.executable,'-m','p2i.cli',*args],text=True,capture_output=True,env=env,timeout=60)
  assert p.returncode==(0 if success else 1),p.stderr+p.stdout
  return json.loads(p.stdout)
 assert cli('skills','list')['success']
 assert cli('skills','search','dropout')['result'][0]['skill_id']=='builtin.dropout'
 assert cli('skills','inspect','builtin.dropout')['result']['status']=='promoted'
 spec=tmp_path/'skill.json';spec.write_text(candidate().model_dump_json())
 source=tmp_path/'skill.py';source.write_text(candidate().implementation.source)
 assert cli('skills','ingest','--spec',str(spec),'--source',str(source))['result']['status']=='candidate'
 assert cli('skills','validate',candidate().id)['result']['retrace_valid']
 assert cli('skills','promote',candidate().id)['result']['status']=='promoted'
 h=p2i.Harness(nn.Sequential(nn.ReLU()),example_inputs=(torch.randn(2,8),))
 arch=tmp_path/'arch.json';h.architecture().save(arch)
 action=tmp_path/'action.json';action.write_text(p2i.ReplaceWithSkill(target='0',skill_id='builtin.gelu').model_dump_json())
 result=cli('harness','apply',str(arch),str(action),'--output',str(tmp_path/'edited.json'),'--input-shape','2','8')
 assert result['success'] and result['committed']
 request=tmp_path/'request.json';request.write_text(json.dumps({'tool':'search_skills','arguments':{'query':'gated'}}))
 assert cli('tool',str(request))['result'][0]['skill_id']==candidate().id
 assert cli('skills','export',str(tmp_path/'portable.json'))['success']
