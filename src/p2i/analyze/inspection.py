"""Opt-in bounded runtime sidecar. Never changes ModelIR's saved schema."""
import math
import torch

class RuntimeInspection:
    def __init__(self,*,max_tensors=64,max_elements=4096,sample_size=16,max_operations=256):
        if not (0<=max_tensors<=256 and 0<=max_elements<=16384 and 0<=sample_size<=64 and 0<=max_operations<=1024):raise ValueError('Inspection limits exceed hard safety caps')
        self.max_tensors=max_tensors;self.max_elements=max_elements;self.sample_size=sample_size;self.max_operations=max_operations
        self.tensors={};self.operations={}
    def tensor(self,info,tensor):
        if len(self.tensors)>=self.max_tensors:return
        entry={'elements':tensor.numel(),'values':None,'message':'Tensor values omitted: size, layout or device threshold.'}
        self.tensors[info.id]=entry
        if tensor.numel()>self.max_elements or tensor.device.type!='cpu' or tensor.layout!=torch.strided or tensor.is_complex() or tensor.is_quantized:return
        try:
            # Capture is observational: its reductions must not become model operations.
            with torch._C._DisableTorchDispatch(),torch.no_grad():
                value=tensor.detach().flatten().float()
                values=value[:self.sample_size].tolist()
                clean=lambda v:v if math.isfinite(v) else None
                entry.update(values=[clean(v) for v in values],message='Bounded sample from this observed execution.')
                if value.numel():entry.update(min=clean(value.min().item()),max=clean(value.max().item()),mean=clean(value.mean().item()),std=clean(value.std(unbiased=False).item()))
        except Exception as exc:entry['message']=f'Values unavailable: {type(exc).__name__}'
    def operation(self,oid,func,args,kwargs):
        if len(self.operations)>=self.max_operations:return
        def compact(v):
            if v is None or type(v) in (bool,int,str):return v
            if type(v) is float:return v if math.isfinite(v) else None
            if isinstance(v,(list,tuple)) and len(v)<=8 and all(type(x) in (bool,int,float,str,type(None)) for x in v):return [compact(x) for x in v]
            raise TypeError
        entry={}
        try:
            for index,spec in enumerate(func._schema.arguments):
                value=args[index] if index<len(args) else kwargs.get(spec.name,spec.default_value)
                try:entry[spec.name]=compact(value)
                except TypeError:pass
        except AttributeError:return
        self.operations[oid]=entry
    def as_dict(self):return {'tensors':self.tensors,'operations':self.operations,'limits':{'max_tensors':self.max_tensors,'max_elements':self.max_elements,'sample_size':self.sample_size}}
