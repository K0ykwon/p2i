import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient
from p2i import Harness
from p2i.server.app import create_app
from p2i.harness.comparison import operation_keys


def test_observed_comparison_keeps_before_until_success():
    h=Harness(nn.Sequential(nn.Linear(4,4),nn.ReLU()),example_inputs=(torch.ones(2,4),))
    h.retrace(capture_values=True)
    client=TestClient(create_app(h))
    assert not client.get('/api/comparison').json()['available']
    assert h.apply({'type':'set_parameter','target':'0','parameter':'bias','value':False}).success
    h.retrace(capture_values=True)
    comparison=client.get('/api/comparison').json()
    assert comparison['before']['revision']==0
    assert comparison['after']['revision']==1
    assert comparison['diff']['parameters']=={'before':20,'after':16}
    assert any(c['path']=='0' for c in comparison['diff']['modules'])
    assert comparison['diff']['tensors']
    assert any(change['after']['grid'] is not None for change in comparison['diff']['tensors'])
    before=h.previous_observation
    # A failed retrace may not publish a new before/after pair.
    h._args=(torch.ones(2,7),)
    with pytest.raises(ValueError):h.retrace()
    assert h.previous_observation is before
    assert h.observed_revision==1


def test_alignment_does_not_use_transient_operation_ids():
    def trace(ids):
        return {'modules':[{'id':'m','qualified_name':'layer'}],'runtime':{'calls':[{'id':'c','sequence':0,'module_id':'m'}]},'operations':[{'id':i,'op_type':kind,'graph':'runtime','call_id':'c'} for i,kind in ids]}
    a=operation_keys(trace([('o0','linear'),('o1','relu')]))
    b=operation_keys(trace([('o9','linear'),('o10','relu')]))
    assert a['o0']==b['o9'] and a['o1']==b['o10']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_cuda_trace_smoke():
    import p2i
    model=nn.Linear(4,4).cuda();ir=p2i.trace(model,example_inputs=(torch.ones(2,4,device='cuda'),))
    assert any(t.device.startswith('cuda') for t in ir.tensors if t.device)
