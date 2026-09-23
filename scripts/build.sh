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
#   vllm             upstream 8e6d8e4f6a + patches/vllm/0001
#                    + drop the unresolvable triton==3.7.2+xpu pin
#                    + host-staged all-reduce through /dev/shm, not gloo
#                      (VLLM_XPU_HOST_STAGED_SHM=1; TP=2 prefill 419 -> 1479 tok/s)
#   vllm-xpu-kernels upstream v0.1.15 + #600 (GDN ragged spec-decode fix,
#                    from main) + patches/vllm-xpu-kernels/0001
#                    + the two source files 0001 references but never ships
#                    (branch humble-b70-v0.1.15; kernels main needs torch
#                    2.14, which vLLM does not support yet)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

VLLM_URL=https://github.com/stolk/vllm.git
VLLM_SHA=0f2e5cf6a38b90eb2dddaa88e7a2738735c92ddf
KERNELS_URL=https://github.com/stolk/vllm-xpu-kernels.git
KERNELS_SHA=789d9a1383b4fd3ee54f7cfe8703a054196708da

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
  "torch==2.13.0+xpu"

# 4. kernels wheel (source build; see docs/drivers.md for toolchain)
#    - oneAPI compiler on PATH (source setvars.sh or install via apt)
#    - the reduced attention presets avoid compiling unused template variants
#      (and their 7-12 GB compiler peaks)
#    - MAX_JOBS defaults to 6; lower it on lower-RAM hosts (fat units can OOM)
export MAX_JOBS="${MAX_JOBS:-6}"
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
# Upstream requirements/xpu.txt pins triton==3.7.2+xpu, Intel's "compatibility
# shim" package, which is not on any reachable index; the fork drops that pin.
# The real package is triton-xpu, on the PyTorch XPU channel at 3.7.2 -- the
# same version the working container ships.
"$HERE/.venv/bin/pip" install --extra-index-url \
  https://download.pytorch.org/whl/xpu "triton-xpu==3.7.2"
"$HERE/.venv/bin/pip" install --no-build-isolation -e src/vllm
"$HERE/.venv/bin/pip" install --force-reinstall --no-deps dist/vllm_xpu_kernels-*.whl

echo "== done. venv at .venv =="
echo "Next: bash scripts/model.sh fetch ; bash scripts/serve.sh"
