from iron.common import AIEContext
from iron.operators.decode_layer_f3best.op import AIEDecodeLayerF3Best
import sys
from pathlib import Path
import os

os.environ["F3BEST_FFN_DIV"] = "1"
os.environ["F3BEST_MT_RELAY"] = "0"
os.environ["F3BEST_MT_DEPTH"] = "2"
os.environ["F3BEST_MT_DECOUPLE"] = "1"

ctx = AIEContext()
op = AIEDecodeLayerF3Best(context=ctx)
op.set_up_artifacts()

from iron.common.compilation import CompilationArtifactGraph
from iron.common.compilation.base import AieccXclbinInstsCompilationRule

graph = CompilationArtifactGraph(op.artifacts)

for rule_cls in ctx.compilation_rules:
    if "AieccXclbinInstsCompilationRule" in rule_cls.__class__.__name__:
        rule = rule_cls
        break

print("PEANO:", rule.peano_dir)
print("AIECC:", rule.aiecc_path)
print("CMD:")
for cmd in rule.compile(graph):
    if hasattr(cmd, 'command'):
        cmds = []
        for c in cmd.command:
            if c.endswith(".mlir"):
                cmds.append("/tmp/f3best_decouple.mlir")
            else:
                cmds.append(c)
        print(" ".join(cmds))
