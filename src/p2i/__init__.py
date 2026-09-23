from .ir import ModelIR
from .analyze.tracer import trace
from .serialization.json import load

from .architecture import ArchitectureIR, build, register_module_adapter, ModuleAdapter, ModuleAdapterRegistry
from .harness import Harness

__version__ = "0.4.0"

def serve(ir, *, host="127.0.0.1", port=8000):
    """Block while serving the local UI and read-only API."""
    import uvicorn
    from .server.app import create_app
    if not isinstance(ir, (ModelIR, Harness)):
        ir = load(ir)
    uvicorn.run(create_app(ir), host=host, port=port)

__all__ = ["ModelIR", "trace", "load", "serve", "Harness", "ArchitectureIR", "build", "register_module_adapter", "ModuleAdapter", "ModuleAdapterRegistry"]

from .skills import SkillRegistry, SkillSpec, TensorContract, ingest_skill, extract_skill, skill_from_subgraph, compare_evaluations
from .actions import ReplaceWithSkill, InsertSkill
__all__ += ['SkillRegistry','SkillSpec','TensorContract','ingest_skill','extract_skill','skill_from_subgraph','compare_evaluations','ReplaceWithSkill','InsertSkill']
