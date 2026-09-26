#!/usr/bin/env bash
# Build the humble-b70 stack from pinned fork commits.
# Produces: venv with vLLM (patched) and the kernels wheel (patched).
#
# Sources come from forks that carry the patches as commits, instead of
# applying patches/ onto upstream. The old apply step ran as
# `git apply ... 2>/dev/null || true`, so a patch that failed to apply was
# silently skipped and only surfaced later as a confusing build error.
# patches/ is kept as the record of the original upstream deltas.
#
#   vllm             upstream main 77871126f9 (2026-09-26, torch 2.14)
#                    + patches/vllm/0001 (host-staged collectives, INT8
#                      lm_head; conflicts with upstream's batch-invariant
#                      collectives resolved, its own mamba_utils pointer fix
#                      dropped as upstream has the same one)
#                    + host-staged all-reduce through /dev/shm, not gloo
#                      (VLLM_XPU_HOST_STAGED_SHM=1; TP=2 prefill 419 -> 1479 tok/s)
#                    (branch humble-b70-next)
#   vllm-xpu-kernels upstream release/0.1.15.4 (what vLLM main pins; has #600)
#                    + patches/vllm-xpu-kernels/0001
#                    + the two source files 0001 references but never ships
#                    + primitive-cache fix for bf16/fp16 INT8 weight scales
#                    (branch humble-b70-v0.1.15.4)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

VLLM_URL=https://github.com/stolk/vllm.git
VLLM_SHA=bfc6e84b14e99a01c7e736fc7feb43e647c3422f
KERNELS_URL=https://github.com/stolk/vllm-xpu-kernels.git
KERNELS_SHA=3a49521f0fbf000aaf9aa0318f095cd1dab571bc

echo "== humble-b70 build =="
echo "vLLM:    $VLLM_SHA"
echo "kernels: $KERNELS_SHA"

# `pip install --user uv` fails on PEP 668 distros (Debian/Ubuntu system
# Python is "externally managed"), so bootstrap uv into a private venv.
if ! command -v uv >/dev/null; then
  [[ -x .uvenv/bin/uv ]] || { python3 -m venv .uvenv && .uvenv/bin/pip install -q uv; }
  export PATH="$HERE/.uvenv/bin:$PATH"
fi

# Check out a pinned commit, cloning or fetching from the fork only if needed.
# Refuses to touch a tree with uncommitted changes to tracked files, rather
# than silently discarding local work.
checkout_pinned() {
  local dir=$1 url=$2 sha=$3
  [[ -d $dir/.git ]] || git clone "$url" "$dir"
  # Fetch every fork branch, so a pin may live on any of them.
  if ! git -C "$dir" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git -C "$dir" fetch "$url" '+refs/heads/*:refs/remotes/fork/*'
  fi
  git -C "$dir" cat-file -e "$sha^{commit}" 2>/dev/null ||
    { echo "ERROR: $sha not found on any branch of $url" >&2; exit 1; }
  if [[ -n $(git -C "$dir" status --porcelain --untracked-files=no) ]]; then
    echo "ERROR: $dir has uncommitted changes; commit or stash them first" >&2
    exit 1
  fi
  # A build dir from another commit is stale: CMake keeps dependency pins such
  # as CUTLASS_REVISION in CMakeCache.txt, so a new checkout would compile
  # against the old sycl-tla headers (seen as "undeclared identifier
  # 'ReduceMode'" after moving the kernels to v0.1.15).
  if [[ $(git -C "$dir" rev-parse HEAD 2>/dev/null) != "$sha" ]]; then
    rm -rf "$dir/build" "$dir/.deps"
  fi
  git -C "$dir" checkout -q --detach "$sha"
}

# 1-2. vLLM and kernels sources at the pinned fork commits
checkout_pinned src/vllm "$VLLM_URL" "$VLLM_SHA"
checkout_pinned src/vllm-xpu-kernels "$KERNELS_URL" "$KERNELS_SHA"

# 3. venv + torch (Intel XPU wheels; an extra index is required)
uv venv --seed --clear --python 3.12 .venv  # --seed: newer uv omits pip; --clear: idempotent on rerun
# The Intel index in the original script no longer serves torch at all, and a
# bare "torch==2.13.0" resolves from PyPI as the CUDA build (+cu130, xpu
# unavailable). The XPU wheels now live on PyTorch's own channel.
"$HERE/.venv/bin/pip" install --extra-index-url \
  https://download.pytorch.org/whl/xpu \
  "torch==2.14.0+xpu"

# 4. kernels wheel (source build; see docs/drivers.md for toolchain)
#    - oneAPI compiler on PATH (source setvars.sh or install via apt)
#    - the reduced attention presets avoid compiling unused template variants
#      (and their 7-12 GB compiler peaks)
#    - MAX_JOBS defaults to 12, sized for 128 GB RAM: each chunk_prefill
#      attention unit peaks near 7 GB in the device compiler, and
#      grouped_gemm_xe2.cpp alone reaches 33 GB. 10 jobs peaked at 93 GB.
#      Use about 5 on a 64 GB host.
export MAX_JOBS="${MAX_JOBS:-12}"
#    - GPU targets: only the B70 (Battlemage G31, PCI 0xe223). Upstream also
#      builds for Ponte Vecchio, the B580 (bmg-g21) and Crescent Island
#      (Xe3P: ~150 extra attention units under csrc/xpu/attn/xe_3). None of
#      the humble-b70 changes are architecture-specific. Override these to
#      build for other cards.
export VLLM_XPU_ENABLE_XE3P="${VLLM_XPU_ENABLE_XE3P:-OFF}"
export VLLM_XPU_AOT_DEVICES="${VLLM_XPU_AOT_DEVICES-bmg-g31}"
export VLLM_XPU_XE2_AOT_DEVICES="${VLLM_XPU_XE2_AOT_DEVICES-bmg-g31}"
export VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf
export VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf
# cmake 3.31.8 (original pin) is not on PyPI: the series goes 3.31.6 -> 3.31.10.
# setuptools_rust: needed by a vLLM build dependency now, absent from the
# original list (ModuleNotFoundError during `pip install -e src/vllm`).
"$HERE/.venv/bin/pip" install numpy "cmake==3.31.10" ninja setuptools_rust \
  "setuptools>=77,<80" setuptools-scm wheel build
# setup.py resolves `cmake` from PATH: without the venv first it picks the
# system CMake (4.x here), which fails in FindPython/Support.cmake. Use the
# pinned 3.31.x that was just installed into the venv.
(cd src/vllm-xpu-kernels && PATH="$HERE/.venv/bin:$PATH" \
  "$HERE/.venv/bin/python" setup.py bdist_wheel \
    --dist-dir "$HERE/dist" --py-limited-api=cp38)

# 5. install vLLM (editable) + kernels wheel
# The real Triton for XPU is triton-xpu, on the PyTorch XPU channel; it must
# match the torch release (3.8.0 for torch 2.14).
"$HERE/.venv/bin/pip" install --extra-index-url \
  https://download.pytorch.org/whl/xpu "triton-xpu==3.8.0"
# vLLM's own compile units are light; VLLM_MAX_JOBS lets them use more cores
# than the memory-hungry kernels build above.
# requirements/xpu.txt names wheels.vllm.ai as an extra index (for the
# triton==3.8.0+xpu shim, which just requires triton-xpu), but pip ignores
# index lines when it reads requirements through setup.py. Pass it here.
MAX_JOBS="${VLLM_MAX_JOBS:-$MAX_JOBS}" \
  "$HERE/.venv/bin/pip" install --no-build-isolation \
    --extra-index-url https://wheels.vllm.ai/xpu/ \
    --extra-index-url https://download.pytorch.org/whl/xpu \
    -e src/vllm
"$HERE/.venv/bin/pip" install --force-reinstall --no-deps dist/vllm_xpu_kernels-*.whl

echo "== done. venv at .venv =="
echo "Next: bash scripts/model.sh fetch ; bash scripts/serve.sh"
