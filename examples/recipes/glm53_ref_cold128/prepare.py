"""Materialize a separately named cold-short harness from the frozen reference inputs."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

SOURCES = {
    "decode_255k/overlay/glm53_mk_ep.py": "73c9fdf8aeb651ee6d822f4385cee825f8a867cd1f7e5f93d0f05f878a78ac6c",
    "decode_255k/serve.sh": "fc51c3cf7964ce9be8fca87455828cecbe8ebbc018a84bb28fd3d65565e98b23",
    "decode_255k/verify_registry.py": "a438cfb06035a572569586ef8c5ba1fbe79fe6b4533ca7aa80bcddf12245ffad",
    "decode_255k/offline-plan.txt": "5ee116026e720cdd348a5ed164713dfcbd6edfdf3d69b297c3bb55d6dbea03a5",
    "short128/study.py": "fdb2248afb97311b6bb51b012d3e3bcbd49bd9aec3c8a6fb9c17788810b0f2b5",
    "native-controls/serve.sh": "ed2b6fefe9c377a0d5c0de794a52690a465dc96111a3a84dfec8740ac11a4ec5",
}
PLAN = SOURCES["decode_255k/offline-plan.txt"]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def edit(body, old, new):
    if body.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {old!r}")
    return body.replace(old, new)


def adapt_hook(s):
    s = edit(s, "        self._warm_prefix = WarmPrefixAttach", """        self._reference_cold_short = os.getenv("MK_GLM53_REFERENCE_COLD_SHORT") == "1"
        self._warm_prefix = WarmPrefixAttach""")
    s = edit(s, """        if self._warm_prefix is None:
            return super().observe_native(runner, output)
""", """        if self._warm_prefix is None:
            if not self._reference_cold_short:
                return super().observe_native(runner, output)
            scheduled = self._schedule(runner, output)
            if scheduled is not None:
                rid, start, prompt, count = scheduled
                new = output.scheduled_new_reqs
                if ((start, prompt, count) == (0, 128, 128) and len(new) == 1
                        and new[0].req_id == rid and new[0].num_computed_tokens == 0
                        and runner.requests[rid].sampling_params.max_tokens == 128):
                    super().observe_native(runner, output)
                    if (self.request_id, self.next_position) == (rid, prompt):
                        logger.info("GLM53 reference cold prefill rank=%d request=%s start=%d end=%d prompt=%d",
                                    self.gate.rank, rid, start, start + count, prompt)
                    return
            self._release("outside the reference cold 128/128 contract")
            return
""")
    s = edit(s, """                     f"offline schedule capture disagrees across TP ranks: {rows}")
        else:""", """                     f"offline schedule capture disagrees across TP ranks: {rows}")
            if self._reference_cold_short and not getattr(self, "_reference_cold_schedule_reported", False):
                logger.info("GLM53 reference cold ready rank=%d schedule=%s python_ops=%d",
                            self.gate.rank, signature, len(self.verified_graph.nodes))
                self._reference_cold_schedule_reported = True
        else:""")
    s = edit(s, """    warm_address_mode = os.getenv("MK_GLM53_WARM_ALIGN_CPU_ADDRESSES", "off")
""", """    cold_short = enabled and os.getenv("MK_GLM53_REFERENCE_COLD_SHORT") == "1"
    cold_config = (not cold_short or (not runner.cache_config.enable_prefix_caching
                   and runner.cache_config.mamba_cache_mode == "none" and runner.model_config.enforce_eager
                   and runner.speculative_config is None and runner.scheduler_config.max_num_seqs == 1
                   and getattr(runner.model, "language_model", runner.model).config.num_hidden_layers == 45
                   and os.getenv("MK_GLM53_LAZY_RUNNER_INPUTS") == "off"
                   and os.getenv("MK_GLM53_CPU_METADATA_PREFETCH", "0") == "0"))
    warm_address_mode = os.getenv("MK_GLM53_WARM_ALIGN_CPU_ADDRESSES", "off")
""")
    s = edit(s, 'long_li, combined_li, typed_owner, private_input))', """long_li, combined_li, typed_owner, private_input,
                                   cold_short, cold_config))""")
    s = edit(s, '''    _require(typed_owner == "0" or (enabled and combine and shared and replicated_owner and not single_origin
             and warm_address_mode in ("on", "shadow") and os.getenv("SCHEDULING_MODE") == "static"''', '''    _require(not cold_short or (cold_config and not warm_prefix and not combine and warm_address_mode == "off"
             and deferral_mode == "off" and topology == (8, 1, 1, 1, 1) and capacity == "262080"
             and long_li == "1" and batch_prefix_copies and fuse_tp_admission and typed_owner == private_input == "1"),
             "reference cold-short requires 45L TP8/DP1/EP8, full native metadata, fixed capacity and typed owner")
    _require(typed_owner == "0" or (enabled and shared and replicated_owner and not single_origin
             and ((combine and warm_address_mode in ("on", "shadow")) or cold_short)
             and os.getenv("SCHEDULING_MODE") == "static"''')
    ast.parse(s)
    return s


def prepare(reference, dest):
    reference = reference.resolve(strict=True)
    dest = dest.resolve()
    if dest.exists() or dest.is_relative_to(reference) or reference.is_relative_to(dest):
        raise ValueError("require a new, separate output directory")
    verified = {name: digest(reference/name) for name in SOURCES}
    if verified != SOURCES:
        raise ValueError(f"unexpected reference inputs {verified}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ref-cold-build-", dir=dest.parent) as tmp:
        out=Path(tmp)/"build"
        out.mkdir()
        overlay=reference/"decode_255k/overlay"
        shutil.copytree(overlay, out/"overlay", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                        symlinks=False)
        hook="glm53_mk_ep.py"
        before=(overlay/hook).read_text()
        after=adapt_hook(before)
        (out/"overlay"/hook).write_text(after)
        (out/"hook.diff").write_text("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile="a/overlay/"+hook,
            tofile="b/overlay/"+hook)))
        source_overlay = {str(p.relative_to(overlay)):digest(p) for p in sorted(overlay.rglob('*'))
                          if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"}
        final_overlay = {str(p.relative_to(out/"overlay")):digest(p) for p in sorted((out/"overlay").rglob('*')) if p.is_file()}
        if set(source_overlay)!=set(final_overlay) or [n for n in source_overlay if source_overlay[n]!=final_overlay[n]]!=[hook]:
            raise ValueError("unexpected device or overlay changes")
        (out/"overlay.sha256").write_text("".join(f"{v}  overlay/{k}\n" for k,v in final_overlay.items()))
        mk=(reference/"decode_255k/serve.sh").read_text()
        mk=edit(mk,'warm_address_mode=on','warm_address_mode=off')
        mk=edit(mk,'candidate=$(cd "$(dirname "$0")" && pwd)',
                'cold_root=$(cd "$(dirname "$0")" && pwd)\noriginal_root=$(cat "$cold_root/original-root")\n'
                'candidate="$original_root/decode_255k"')
        mk=edit(mk,'warm_overlay="$candidate/overlay"',
                'warm_overlay="$cold_root/overlay"\n( cd "$cold_root"; sha256sum --check --strict overlay.sha256 ) > cold-overlay-check.log')
        mk=edit(mk,'MK_GLM53_WARM_ALIGN_LI_COMBINED=1','MK_GLM53_WARM_ALIGN_LI_COMBINED=0')
        mk=edit(mk,'MK_GLM53_WARM_PREFIX_ATTACH=1','MK_GLM53_WARM_PREFIX_ATTACH=0')
        mk=edit(mk,'ep_port=$(python3',"""export MK_GLM53_REFERENCE_COLD_SHORT=1 MK_GLM53_LAZY_RUNNER_INPUTS=off
case "$GLM53_REF_COLD_PLAN" in
    required)
        [[ "$MK_GLM53_OFFLINE_STATIC_COSTS" == "$candidate/offline-plan.txt" ]]
        printf '%s  %s\\n' 'PLAN_SHA' "$MK_GLM53_OFFLINE_STATIC_COSTS" | sha256sum --strict -c - ;;
    stock) unset MK_GLM53_OFFLINE_STATIC_COSTS ;;
    *) echo 'Explicit GLM53_REF_COLD_PLAN=required or stock required' >&2; exit 2 ;;
esac
printf '%s\\n' "GLM53_REF_COLD_PLAN=$GLM53_REF_COLD_PLAN" "MK_GLM53_OFFLINE_STATIC_COSTS=PLAN_VALUE" \\
    "MK_GLM53_LAZY_RUNNER_INPUTS=$MK_GLM53_LAZY_RUNNER_INPUTS" \\
    "MK_GLM53_REFERENCE_COLD_SHORT=$MK_GLM53_REFERENCE_COLD_SHORT" > cold-env.txt
ep_port=$(python3""".replace("PLAN_SHA",PLAN).replace("PLAN_VALUE",'$'+'{MK_GLM53_OFFLINE_STATIC_COSTS:-}'))
        mk=edit(mk,'--enable-prefix-caching','--no-enable-prefix-caching')
        (out/"serve-mk.sh").write_text(mk)
        native=edit((reference/"native-controls/serve.sh").read_text(),'--enable-prefix-caching','--no-enable-prefix-caching')
        (out/"serve-native.sh").write_text(native)
        (out/"original-root").write_text(str(reference)+"\n")
        study=(reference/"short128/study.py").read_text()
        study=edit(study,'ROOT=Path(__file__).resolve().parents[1]',
                   "COLD=Path(__file__).resolve().parent\nROOT=Path((COLD/'original-root').read_text().strip())")
        study=edit(study,"str(reference/'serve.sh')","str(COLD/'serve-mk.sh')")
        study=edit(study,"ROOT/'native-controls/serve.sh'","COLD/'serve-native.sh'")
        (out/"short-study.py").write_text(study)
        runner=(reference/"short128/run.sh").read_text()
        runner=edit(runner,'r=$(cd "$here/.." && pwd)','r=$(cat "$here/original-root")')
        runner=edit(runner,'"$here/study.py"','"$here/short-study.py"')
        (out/"inner.sh").write_text(runner)
        launcher=(reference/"with-reference-system-curl.sh").read_text()
        launcher=edit(launcher,'here=$(cd "$(dirname "$0")" && pwd)',
                      'cold_root=$(cd "$(dirname "$0")" && pwd)\nhere=$(cat "$cold_root/original-root")')
        launcher=edit(launcher,'$mode == native || $mode == mk || $mode == short-native || $mode == short-mk',
                      '$mode == native || $mode == mk')
        launcher=edit(launcher,'short-native|short-mk) shift;exec bash "$here/short128/run.sh" run "MODE" "$@";;'.replace("MODE",'$'+'{mode#short-}'),
                      'native|mk) exec bash "$cold_root/controller.sh" "$@";;')
        launcher=edit(launcher,'    *) exec bash "$here/with-reference-vendor.sh" "$@";;','    *) exit 2;;')
        (out/"run.sh").write_text(launcher)
        (out/"controller.sh").write_text("""#!/usr/bin/env bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
[[ "$#" = 3 && ( "$1" = native || "$1" = mk ) && ( "$3" = required || "$3" = stock ) ]]
export GLM53_REF_COLD_PLAN="$3"
if bash "$here/inner.sh" run "$1" "$2"; then rc=0; else rc=$?; fi
# Always retain a classified failure if the server or original plan refuses the shape.
if /usr/local/python3.11.10/bin/python3 -IB "$here/audit.py" arm "$1" "$2" "$3"; then ar=0; else ar=$?; fi
if [[ $rc -ne 0 ]]; then exit "$rc"; fi
exit "$ar"
""")
        shutil.copyfile(Path(__file__).with_name("audit.py"),out/"audit.py")
        for p in out.glob('*.sh'):p.chmod(0o755)
        inputs={str(p.relative_to(out)):digest(p) for p in sorted(out.iterdir()) if p.is_file()}
        (out/"manifest.json").write_text(json.dumps(dict(
           version="cold-short-1",reference_git="2d4f96473945a4a1c4149d0360eeb8b64e848369",
           sources=verified,original_overlay=source_overlay,adapted_overlay=final_overlay,files=inputs),
           indent=2,sort_keys=True)+"\n")
        out.rename(dest)
    return {"output":str(dest), "manifest":digest(dest/"manifest.json"), "overlay":digest(dest/"overlay"/"glm53_mk_ep.py"),"patch":digest(dest/"hook.diff")}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("reference",type=Path)
    p.add_argument("new_output",type=Path)
    a=p.parse_args()
    print(json.dumps(prepare(a.reference,a.new_output)))
