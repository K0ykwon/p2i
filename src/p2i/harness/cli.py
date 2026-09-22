"""Stateless JSON artifact commands. Never load pickle or arbitrary Python files."""
import json
from pathlib import Path
import torch
from p2i.architecture import ArchitectureIR,architecture_diff
from .core import Harness

def add_commands(sub):
    root=sub.add_parser('harness',help='Operate on architecture/action JSON artifacts')
    commands=root.add_subparsers(dest='harness_command',required=True)
    for command in ('inspect','apply','validate'):
        p=commands.add_parser(command);p.add_argument('architecture');p.add_argument('--registry');p.add_argument('--allow-python',action='store_true')
        if command=='apply':p.add_argument('action');p.add_argument('--output',required=True);p.add_argument('--preview',action='store_true')
        if command in ('apply','validate'):p.add_argument('--input-shape',type=int,nargs='+',help='Optional one float32 example input shape')
    p=commands.add_parser('diff');p.add_argument('before');p.add_argument('after')

def run(args):
    try:
        if args.harness_command=='diff':result=architecture_diff(ArchitectureIR.load(args.before),ArchitectureIR.load(args.after)).model_dump(mode='json');success=True
        else:
            arch=ArchitectureIR.load(args.architecture)
            inputs=(torch.zeros(*args.input_shape),) if getattr(args,'input_shape',None) else None
            from p2i.skills import SkillRegistry
            h=Harness.from_architecture(arch,example_inputs=inputs,skill_registry=SkillRegistry(args.registry,allow_python=args.allow_python))
            if args.harness_command=='inspect':result={'summary':h.observe(),'architecture':h.observe(level='architecture')};success=True
            elif args.harness_command=='validate':
                report=h.validate();result=report.model_dump(mode='json');success=report.valid
            else:
                action=json.loads(Path(args.action).read_text())
                result_object=h.preview(action) if args.preview else h.apply(action)
                result=result_object.model_dump(mode='json');success=result_object.success
                if success and not args.preview:h.architecture().save(args.output)
        print(json.dumps(result,sort_keys=True));return 0 if success else 1
    except Exception as exc:
        print(json.dumps({'success':False,'errors':[str(exc)]}));return 1
