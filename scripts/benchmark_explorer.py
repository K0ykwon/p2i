"""Reproducible CPU runtime-only graph fixtures and measurements (not a leaderboard)."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path
import torch
from torch import nn
import p2i
from p2i.analyze.inspection import RuntimeInspection


class Chain(nn.Module):
    def __init__(self,count):
        super().__init__();self.count=count
    def forward(self,x):
        for _ in range(self.count):x=torch.sin(x)
        return x


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sizes',type=int,nargs='+',default=[500,1000,3000]);parser.add_argument('--output',default='artifacts/performance');args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);torch.set_num_threads(1)
    records=[]
    for size in args.sizes:
        model=Chain(size);x=torch.randn(2,8)
        # Warm initialization before measuring; retain no tensors in JSON.
        p2i.trace(Chain(1),example_inputs=(x,),static=False)
        for capture in (False,True):
            inspection=RuntimeInspection() if capture else None
            start=time.perf_counter();ir=p2i.trace(model,example_inputs=(x,),static=False,inspection=inspection);elapsed=time.perf_counter()-start
            encoded=ir.model_dump_json();records.append({'requested_operations':size,'operations':len(ir.operations),'capture':capture,'trace_seconds':elapsed,'ir_bytes':len(encoded.encode()),'inspection_bytes':len(json.dumps(inspection.as_dict())) if inspection else 0,'process_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            if not capture:(output/f'graph-{size}.json').write_text(encoded)
    result={'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'platform':platform.platform(),'device':'cpu','threads':1,'seed':0},'notes':'Single measured pass after warmup; max RSS is cumulative process peak on Linux; static tracing excluded. Repeat before drawing conclusions.','measurements':records}
    (output/'trace.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
