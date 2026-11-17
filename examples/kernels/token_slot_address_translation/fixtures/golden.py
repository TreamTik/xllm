# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Task-provided fixtures; reference source strings remain unchanged.

from typing import Any

"CPU 标杆与 NPU 基线接口。"
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

TASK_NUMBER = 5
API_EXPORTS = {"calculate": ["public_api", "calculate"]}
REFERENCE_SOURCES = {
    "public_api.py": "\"\"\"Independent Python scalar oracle; input/output are lists, never the candidate.\"\"\"\nTASK = 'slot'\ndef calculate(values, attrs, single=False):\n    if TASK == 'flags':\n        offsets,indptr,lens,tileptr=values; tile=attrs['tile_size']; mx=attrs['max_tiles_per_seq']\n        result=[0]*(len(lens)*mx)\n        for r,length in enumerate(lens):\n            for t in range(mx):\n                begin=t*tile\n                if begin+tile>length: continue\n                data=offsets[indptr[r]+begin:indptr[r]+begin+tile]\n                result[tileptr[r]+t]=int(data[0]%tile==0 and all(data[i]-data[i-1]==1 for i in range(1,tile)))\n        return result\n    x,m=values[:2]; rows=[m] if single else m\n    outputs=[]\n    for row in rows:\n        if TASK=='dense': result=[[row[v] if v>=0 else -1 for v in req] for req in x]\n        elif TASK=='slot':\n            b=attrs['block_size']; result=[row[v//b]*b+v%b if v>=0 else -1 for v in x]\n        else:\n            cu=values[2]; result=[]\n            for r,req in enumerate(x): result.extend(row[v] if v>=0 else -1 for v in req[:cu[r+1]-cu[r]])\n        outputs.append(result)\n    return outputs[0] if single else outputs\n",
    "baseline.py": "\"\"\"Ordinary PyTorch NPU vector operations, frozen comparison baseline.\"\"\"\nimport torch\nTASK = 'slot'\ndef prepare(inputs, attrs, info):\n    if TASK=='packed':\n        x=inputs[0]; cu=info['values'][2]\n        # The original caller supplies counts. Build layout once, outside both hot paths.\n        ids=[r*x.shape[1]+c for r in range(x.shape[0]) for c in range(cu[r+1]-cu[r])]\n        ids=torch.tensor(ids,dtype=torch.int64,device=x.device)\n    elif TASK=='flags':\n        tile=attrs['tile_size']; mx=attrs['max_tiles_per_seq']; rows=info['rows']\n        tile_ids=torch.arange(mx,device=inputs[0].device).reshape(1,mx,1)\n        lane=torch.arange(tile,device=inputs[0].device).reshape(1,1,tile)\n        pos=tile_ids*tile+lane\n    def run():\n        if TASK in ('dense','slot','packed'):\n            x,m=inputs[:2]; m=m.reshape(info['layers'],info['blocks'])\n            if TASK=='packed': x=x.reshape(-1).index_select(0,ids)\n            flat=x.reshape(-1).to(torch.int64)\n            if TASK=='slot':\n                b=attrs['block_size']; safe=flat.clamp_min(0); logical=safe//b\n                result=m.index_select(1,logical)*b+(safe%b).reshape(1,-1)\n            else: result=m.index_select(1,flat.clamp_min(0))\n            result=torch.where(flat.reshape(1,-1)<0,-1,result).to(torch.int32)\n            shape=tuple(x.shape) if info['single'] else (info['layers'],*x.shape)\n            return result.reshape(shape)\n        offsets,ip,lens,tp=inputs\n        if rows==0: return torch.empty((0,),dtype=torch.uint8,device=offsets.device)\n        if offsets.numel()==0: return torch.zeros((rows*mx,),dtype=torch.uint8,device=offsets.device)\n        address=ip[:-1].to(torch.int64).reshape(rows,1,1)+pos\n        data=offsets.index_select(0,address.clamp_max(offsets.numel()-1).reshape(-1)).reshape(rows,mx,tile)\n        first=data[:,:,:1]; consecutive=(data==first+lane).all(dim=-1)\n        full=(tile_ids.reshape(1,mx)+1)*tile<=lens.reshape(rows,1)\n        return (consecutive & (first.squeeze(-1)%tile==0) & full).to(torch.uint8).reshape(-1)\n    return run\n",
    "contract.py": '"""Metadata/value validation occurs in prepare; kernels never calculate on CPU."""\nimport torch\nTASK = \'slot\'\nI32 = 2**31-1\n\ndef validate(inputs, attributes):\n    if not isinstance(inputs, (tuple,list)) or not isinstance(attributes,dict):\n        raise TypeError("inputs must be sequence; attributes must be dict")\n    expected = 4 if TASK == \'flags\' else 3 if TASK == \'packed\' else 2\n    if len(inputs) != expected: raise ValueError("input count")\n    allowed = {\'block_size\'} if TASK == \'slot\' else {\'tile_size\',\'max_tiles_per_seq\'} if TASK == \'flags\' else set()\n    if set(attributes) != allowed: raise ValueError("attributes missing or unknown")\n    for value in attributes.values():\n        if type(value) is not int or value <= 0: raise ValueError("attributes must be positive integers")\n    for x in inputs:\n        if not isinstance(x,torch.Tensor) or x.dtype != torch.int32: raise TypeError("all inputs int32")\n        if x.device.type != \'npu\': raise ValueError("NPU inputs required")\n        if not x.is_contiguous(): raise ValueError("contiguous inputs required; storage offset is supported")\n        if x.device != inputs[0].device: raise ValueError("same device required")\n        if x.numel() > I32: raise ValueError("input indexing exceeds int32")\n    # These readbacks are a one-time preflight, excluded from prepared hot-path timing.\n    values = [x.detach().cpu().tolist() for x in inputs]\n    if TASK in (\'dense\',\'slot\',\'packed\'):\n        x,m = inputs[:2]; xv,mv = values[:2]\n        if m.ndim not in (1,2) or m.shape[-1] < 1: raise ValueError("M shape [B] or [L,B], B>=1")\n        if m.ndim == 2 and m.shape[0] < 1: raise ValueError("L>=1")\n        rows = [mv] if m.ndim == 1 else mv\n        if any(v<0 for row in rows for v in row): raise ValueError("M must be nonnegative")\n        if TASK == \'slot\':\n            if x.ndim != 1: raise ValueError("slots shape [N]")\n            b=attributes[\'block_size\']\n            if b>I32: raise ValueError("block_size int32 overflow")\n            if any(v>=0 and v//b>=m.shape[-1] for v in xv): raise ValueError("logical slot out of range")\n            if any(v*b+b-1>I32 for row in rows for v in row): raise ValueError("translated slot int32 overflow")\n        else:\n            if x.ndim != 2: raise ValueError("logical table shape [R,C]")\n            if TASK == \'packed\':\n                cu=inputs[2]; cv=values[2]\n                if cu.ndim!=1 or len(cv)!=x.shape[0]+1 or cv[0]!=0: raise ValueError("invalid cu shape/start")\n                if any(cv[i]>cv[i+1] or cv[i+1]-cv[i]>x.shape[1] for i in range(x.shape[0])): raise ValueError("invalid cu counts")\n                active=[row[:cv[r+1]-cv[r]] for r,row in enumerate(xv)]\n            else: active=xv\n            if any(v>=m.shape[-1] for row in active for v in row): raise ValueError("logical block out of range")\n        output_count = len(rows) * (values[2][-1] if TASK == \'packed\' else x.numel())\n        if output_count > I32: raise ValueError("output indexing exceeds int32")\n        return {\'values\':values,\'layers\':len(rows),\'blocks\':m.shape[-1], \'single\':m.ndim==1}\n    offsets,indptr,lens,tileptr=inputs; ov,iv,lv,tv=values\n    if any(x.ndim!=1 for x in inputs): raise ValueError("flags inputs must be vectors")\n    if len(iv)!=len(lv)+1 or len(tv)!=len(lv)+1 or iv[0]!=0: raise ValueError("indptr lengths/start")\n    tile=attributes[\'tile_size\']; mx=attributes[\'max_tiles_per_seq\']\n    if tile>4096: raise ValueError("tile_size supported range 1..4096")\n    if len(lv)*mx > I32: raise ValueError("output indexing exceeds int32")\n    if iv[-1]!=len(ov) or any(iv[i]>iv[i+1] or lv[i]<0 or lv[i]>iv[i+1]-iv[i] for i in range(len(lv))): raise ValueError("offset segments/lens")\n    if tv != [i*mx for i in range(len(lv)+1)]: raise ValueError("dense tile output required")\n    if any((n+tile-1)//tile>mx for n in lv) or any(v<0 for v in ov): raise ValueError("capacity or offsets")\n    return {\'values\':values,\'rows\':len(lv)}\n',
}
_SESSION = None


def _write_sources(root: Any, sources: Any) -> Any:
    for name, source in sources.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid embedded source path")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


class _ImportNames(ast.NodeTransformer):
    def __init__(self, prefix: Any, names: Any) -> Any:
        self.prefix, self.names = (prefix, names)

    def visit_Import(self, node: Any) -> Any:
        result = []
        for alias in node.names:
            if alias.name.split(".")[0] in self.names:
                if "." in alias.name:
                    result.append(
                        ast.Import(
                            names=[ast.alias(self.prefix + "." + alias.name, alias.asname or alias.name.split(".")[0])]
                        )
                    )
                else:
                    result.append(
                        ast.ImportFrom(module=self.prefix, names=[ast.alias(alias.name, alias.asname)], level=0)
                    )
            else:
                result.append(ast.Import(names=[alias]))
        return [ast.copy_location(x, node) for x in result]

    def visit_ImportFrom(self, node: Any) -> Any:
        if not node.level and node.module and (node.module.split(".")[0] in self.names):
            node.module = self.prefix + "." + node.module
        return node


class _ReferenceLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Isolate absolute imports; function bodies and native source stay unchanged."""

    def __init__(self, root: Any) -> Any:
        self.root = root
        self.prefix = "_task_reference_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        self.names = {p.stem for p in root.glob("*.py")}
        package = types.ModuleType(self.prefix)
        package.__path__ = [str(root)]
        sys.modules[self.prefix] = package
        sys.meta_path.insert(0, self)

    def find_spec(self, fullname: Any, path: Any = None, target: Any = None) -> Any:
        if fullname.startswith(self.prefix + "."):
            name = fullname[len(self.prefix) + 1 :]
            source = self.root / (name.replace(".", "/") + ".py")
            if source.is_file():
                return importlib.util.spec_from_loader(fullname, self, origin=str(source))
        return None

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> Any:
        path = Path(module.__spec__.origin)
        module.__file__ = str(path)
        tree = ast.parse(path.read_text(), str(path))
        tree = ast.fix_missing_locations(_ImportNames(self.prefix, self.names).visit(tree))
        exec(compile(tree, str(path), "exec"), module.__dict__)

    def get(self, name: Any) -> Any:
        return importlib.import_module(self.prefix + "." + name)


def reference(name: Any) -> Any:
    """Load a frozen oracle/baseline by its documented module name."""
    global _SESSION
    if _SESSION is None:
        temp = tempfile.TemporaryDirectory(prefix="task-reference-")
        root = Path(temp.name)
        _write_sources(root, REFERENCE_SOURCES)
        _SESSION = (temp, _ReferenceLoader(root))
    return _SESSION[1].get(name)


def __getattr__(name: Any) -> Any:
    if name in API_EXPORTS:
        module, attribute = API_EXPORTS[name]
        return getattr(reference(module), attribute)
    raise AttributeError(name)


def main() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description='标杆 calculate(values, attrs, single=False)；基线 reference("baseline").prepare(inputs, attrs, info)'
    )
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
