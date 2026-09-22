import argparse
import sys
import p2i

def main():
    parser = argparse.ArgumentParser(prog="p2i", description="Local PyTorch model explorer")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Serve a saved Model IR")
    serve.add_argument("file")
    demo = sub.add_parser("demo", help="Trace an offline bundled model and serve it")
    demo.add_argument("name", choices=["mlp", "cnn", "transformer"])
    demo.add_argument("--save", default=None, help="Also write the IR to this JSON path")
    demo.add_argument("--no-serve", action="store_true", help="Only analyze (use with --save)")
    for p in (serve, demo):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000)
    demo.add_argument("--edit", action="store_true", help="Enable a live architecture Harness in the local UI")
    from .harness.cli import add_commands, run
    add_commands(sub)
    from .skills.cli import add_commands as add_skill_commands, run as run_skills
    add_skill_commands(sub)
    args = parser.parse_args()
    if args.command in ("skills", "tool"):
        raise SystemExit(run_skills(args))
    if args.command == "harness":
        raise SystemExit(run(args))
    try:
        if args.command == "serve":
            ir = p2i.load(args.file)
        else:
            from .examples import make_demo
            model, inputs = make_demo(args.name)
            ir = p2i.trace(model, example_inputs=inputs)
            if args.save:
                ir.save(args.save)
            print(f"{ir.metadata.name}: {len(ir.modules)} modules, {len(ir.operations)} operations, {len(ir.tensors)} tensors")
            for result in ir.metadata.analysis:
                print(f"  {result.technique}: {result.status}")
            if args.no_serve:
                return
        if args.command == "demo" and args.edit:
            ir = p2i.Harness(model, ir=ir, example_inputs=inputs)
        p2i.serve(ir, host=args.host, port=args.port)
    except (ValueError, OSError) as exc:
        print(f"p2i: {exc}", file=sys.stderr)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
