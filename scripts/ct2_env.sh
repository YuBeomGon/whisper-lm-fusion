#!/usr/bin/env bash
# Local dev convenience: point Python at an ALREADY-BUILT patched CTranslate2
# checkout (KenLM BPE fusion) without installing it.
#
# Canonical install is from the fork, built from source (see README "KenLM fusion"):
#   github.com/YuBeomGon/CTranslate2  branch feature/kenlm-bpe-fusion
#
# This shim exists because the pip-installed ctranslate2 binding is ABI-incompatible
# with the patched libctranslate2 here, and anaconda's libstdc++ is too old. It just
# preloads the system libstdc++ + the locally-built patched lib and binding.
#
# Usage:  source scripts/ct2_env.sh   (then run python / pytest as usual)

_CT2_ROOT="${CT2_ROOT:-../CTranslate2}"
_CT2_BINDING="${_CT2_ROOT}/python/build/lib.linux-x86_64-cpython-312"
_CT2_LIB="${_CT2_ROOT}/build-kenlm/libctranslate2.so.4.7.2"
_SYS_STDCXX="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

export PYTHONPATH="${_CT2_BINDING}${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD="${_SYS_STDCXX} ${_CT2_LIB}${LD_PRELOAD:+ $LD_PRELOAD}"

echo "patched CTranslate2 env set:"
echo "  PYTHONPATH += ${_CT2_BINDING}"
echo "  LD_PRELOAD  = ${LD_PRELOAD}"
