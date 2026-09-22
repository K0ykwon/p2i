import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient
import p2i
from p2i.examples import make_demo
from p2i.analyze import export,fx
from p2i.server.app import create_app

@pytest.mark.parametrize('name,shape', [('mlp',[2,4]),('cnn',[2,5]),('transformer',[2,16,8])])
def test_models(name,shape,tmp_path):
    model,args=make_demo(name)
    ir=p2i.trace(model,example_inputs=args)
    assert ir.modules and ir.operations and ir.tensors
    assert all(a.status=='success' for a in ir.metadata.analysis if a.technique in ('export','runtime_hooks','runtime_dispatch'))
    assert {o.graph for o in ir.operations}=={'runtime','export'}
    assert next(t.shape for t in ir.tensors if t.id==ir.runtime.output_tensor_ids[0])==shape
    assert all(o.parent_module_id is not None for o in ir.operations if o.graph=='runtime')
    ir.save(tmp_path/'model.json')
    client=TestClient(create_app(p2i.load(tmp_path/'model.json')))
    assert client.get('/api/model').json()['metadata']['name']==type(model).__name__
    for kind in ('modules','operations','tensors'):
        assert client.get(f'/api/{kind}/{getattr(ir,kind)[0].id}').status_code==200
    assert client.get('/api/modules/invalid').status_code==404
    assert client.get('/').status_code==200
    assert '<div id="root">' in client.get('/').text

def fail(*args,**kwargs): raise RuntimeError('deliberate backend failure')

def test_both_static_failures(monkeypatch):
    monkeypatch.setattr(export,'capture',fail)
    monkeypatch.setattr(fx,'capture',fail)
    ir=p2i.trace(nn.Linear(4,2),(torch.ones(1,4),))
    assert len(ir.operations)>0
    assert [a.status for a in ir.metadata.analysis[-2:]]==['failed','failed']
    assert all(o.graph=='runtime' for o in ir.operations)

def test_fx_fallback(monkeypatch):
    monkeypatch.setattr(export,'capture',fail)
    ir=p2i.trace(nn.Sequential(nn.Linear(4,2),nn.ReLU()),(torch.ones(1,4),))
    assert ir.metadata.analysis[-1].status=='success'
    assert any(o.graph=='fx' for o in ir.operations)
    assert any(t.graph=='fx' and t.shape==[1,2] for t in ir.tensors)
    assert any(e.source_id.startswith('f') for e in ir.data_edges)

def test_data_dependent_control_flow():
    class Branch(nn.Module):
        def forward(self,x):
            if x.sum()>0: return x.sin()
            return x.cos()
    ir=p2i.trace(Branch(),(torch.ones(2),))
    assert any('sin' in o.op_type for o in ir.operations if o.graph=='runtime')
    assert not any('cos' in o.op_type for o in ir.operations if o.graph=='runtime')
    assert next(a.status for a in ir.metadata.analysis if a.technique=='fx')=='failed'

def test_forward_failure_cleanup():
    class Broken(nn.Module):
        def forward(self,x):
            y=x+1
            raise ValueError('intentional failure')
    model=Broken()
    ir=p2i.trace(model,(torch.ones(2),),static=False,isolate=False)
    assert ir.metadata.warnings
    assert ir.operations
    assert not ir.runtime.calls[0].completed
    assert not model._forward_hooks and not model._forward_pre_hooks

def test_library_attention():
    class UsesLibrary(nn.Module):
        def __init__(self):
            super().__init__();self.attention=nn.MultiheadAttention(16,4,batch_first=True)
        def forward(self,x):return self.attention(x,x,x,need_weights=False)[0]
    ir=p2i.trace(UsesLibrary().eval(),(torch.randn(2,5,16),))
    attention=next(m for m in ir.modules if m.name=='attention')
    assert attention.origin.kind=='framework' and attention.executed
    assert attention.children
    assert any(o.parent_module_id==attention.id for o in ir.operations if o.graph=='runtime')

def test_fx_partial_shape_failure(monkeypatch):
    from torch.fx.passes.shape_prop import ShapeProp
    monkeypatch.setattr(export,'capture',fail)
    monkeypatch.setattr(ShapeProp,'propagate',fail)
    ir=p2i.trace(nn.Sequential(nn.Linear(4,2),nn.ReLU()),(torch.ones(1,4),))
    assert ir.metadata.analysis[-1].status=='partial'
    assert any(o.graph=='fx' for o in ir.operations)
    assert any(o.graph=='runtime' for o in ir.operations)
