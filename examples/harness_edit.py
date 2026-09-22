"""Offline Phase 2 acceptance: explicit edits, validation, weights and new evidence."""
from pathlib import Path
import json
import p2i
from p2i.examples import make_demo

def run(name):
 model,args=make_demo(name)
 h=p2i.Harness(model,example_inputs=args)
 before=h.retrace()
 if name in ('mlp','transformer'):
  first,last,width=('layers.0','layers.2',24) if name=='mlp' else ('blocks.0.mlp.0','blocks.0.mlp.2',96)
  action={'type':'batch','expected_revision':0,'actions':[
   {'type':'set_parameter','target':first,'parameter':'out_features','value':width},
   {'type':'set_parameter','target':last,'parameter':'in_features','value':width}]}
 else:
  rejected=h.apply({'type':'set_parameter','target':'features.3','parameter':'out_channels','value':24})
  assert not rejected.success
  action={'type':'set_parameter','target':'features.0','parameter':'padding_mode','value':'reflect','expected_revision':0}
 result=h.apply(action)
 assert result.success,result.errors
 assert h.validate(backward=True).backward_valid
 model=h.build();model(*args)
 after=h.retrace(capture_values=True)
 out=Path('artifacts/phase2');out.mkdir(parents=True,exist_ok=True)
 h.architecture().save(out/f'{name}.architecture.json')
 after.save(out/f'{name}.observed.json')
 (out/f'{name}.action.json').write_text(json.dumps(action,indent=2))
 (out/f'{name}.report.json').write_text(h.observe_json())
 print(f'{name}: revision {h.observe()["revision"]}, forward/backward/retrace passed, parameters {before.modules[0].parameter_count} -> {after.modules[0].parameter_count}')
 return h

if __name__=='__main__':
 for name in ('mlp','cnn','transformer'):run(name)
