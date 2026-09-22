import json
from pathlib import Path
from .registry import SkillRegistry
from .schema import SkillSpec,SkillError

def add_commands(sub):
    root=sub.add_parser('skills',help='Local skills and candidate validation (JSON output)')
    root.add_argument('--registry',default=None,help='SQLite path; defaults to P2I_SKILL_REGISTRY or ~/.p2i/skills.sqlite3')
    commands=root.add_subparsers(dest='skills_command',required=True)
    for name in ('list','search','inspect','validate','promote','reject','deprecate','ingest','export','import'):
        p=commands.add_parser(name)
        if name=='search':p.add_argument('query')
        if name in ('list','search'):p.add_argument('--status',default=None)
        if name in ('inspect','validate','promote','reject','deprecate'):p.add_argument('skill_id')
        if name=='inspect':p.add_argument('--versions',action='store_true')
        if name=='validate':
            p.add_argument('--parameters',default='{}');p.add_argument('--dimensions',default='{}');p.add_argument('--timeout',type=float,default=30)
        if name=='ingest':p.add_argument('--spec',required=True);p.add_argument('--source')
        if name in ('export','import'):p.add_argument('file')
    tool=sub.add_parser('tool',help='Invoke a provider-independent JSON tool')
    tool.add_argument('request',nargs='?');tool.add_argument('--schemas',action='store_true');tool.add_argument('--registry');tool.add_argument('--architecture');tool.add_argument('--output');tool.add_argument('--input-shape',type=int,nargs='+');tool.add_argument('--allow-python',action='store_true',help='Explicitly trust promoted Python skills for in-process model construction')

def run(args):
    try:
        if args.command=='tool':
            from p2i.tools import dispatch,tool_schemas
            if args.schemas:print(json.dumps(tool_schemas(),sort_keys=True));return 0
            registry=SkillRegistry(args.registry,allow_python=args.allow_python)
            h=None
            if args.architecture:
                import torch
                from p2i import Harness,ArchitectureIR
                h=Harness.from_architecture(ArchitectureIR.load(args.architecture),example_inputs=(torch.zeros(args.input_shape),) if args.input_shape else None,skill_registry=registry)
            out=dispatch(json.loads(Path(args.request).read_text()),harness=h,registry=registry)
            if out.success and h and args.output:h.architecture().save(args.output)
            print(out.model_dump_json());return 0 if out.success else 1
        r=SkillRegistry(args.registry);name=args.skills_command
        if name=='list':out=r.list(status=args.status)
        elif name=='search':out=r.search(query=args.query,status=args.status)
        elif name=='inspect':out=r.versions(args.skill_id) if args.versions else r.get(args.skill_id)
        elif name=='validate':out=r.validate(args.skill_id,parameters=json.loads(args.parameters),dimensions=json.loads(args.dimensions),timeout=args.timeout)
        elif name in ('promote','reject','deprecate'):out=getattr(r,name)(args.skill_id)
        elif name=='ingest':
            from . import ingest_skill
            spec=SkillSpec.model_validate_json(Path(args.spec).read_text())
            out=ingest_skill(source=Path(args.source).read_text(),metadata=spec,registry=r) if args.source else r.register(spec)
        elif name=='export':r.export(args.file);out={'file':args.file}
        elif name=='import':out=r.import_file(args.file)
        success=getattr(out,'valid',True)
        data=out.model_dump(mode='json') if hasattr(out,'model_dump') else [s.model_dump(mode='json') for s in out] if isinstance(out,list) else out
        print(json.dumps({'success':success,'result':data},sort_keys=True));return 0 if success else 1
    except Exception as exc:
        print(json.dumps({'success':False,'error':{'code':getattr(exc,'code','EXECUTION_FAILED'),'message':str(exc)}}));return 1
