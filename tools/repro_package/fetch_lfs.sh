#!/usr/bin/env bash
# Fetch the git-LFS netCDF models of the IntervalMDP.jl reproducibility package without git-lfs.
#
#   REPRO_PACKAGE=<checkout> tools/repro_package/fetch_lfs.sh <out_dir> [model ...]
#
# The repository (github.com/Zinoex/IntervalMDP.jl_ReproducibilityPackage) is not public, so the
# anonymous media URL 404s; this asks GitHub for an LFS token over SSH (needs a key with read
# access), resolves each object through the LFS batch API, and checks its sha256 against the
# pointer file in the local checkout.
set -euo pipefail

REPO="Zinoex/IntervalMDP.jl_ReproducibilityPackage"
POINTERS="${REPRO_PACKAGE:?set REPRO_PACKAGE to a checkout of IntervalMDP.jl_ReproducibilityPackage (its LFS pointer files)}/data"
OUT="${1:?usage: fetch_lfs.sh <out_dir> [model ...]}"; shift
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(multiObj_robotIMDP pimdp_0 pimdp_1 pimdp_2)

mkdir -p "$OUT"
AUTH=$(ssh -o BatchMode=yes git@github.com git-lfs-authenticate "$REPO" download |
       python3 -c 'import json,sys; print(json.load(sys.stdin)["header"]["Authorization"])')

for m in "${MODELS[@]}"; do
    oid=$(grep '^oid' "$POINTERS/$m.nc" | cut -d: -f2)
    size=$(grep '^size' "$POINTERS/$m.nc" | cut -d' ' -f2)
    href=$(curl -sS -X POST \
        -H "Authorization: $AUTH" \
        -H "Accept: application/vnd.git-lfs+json" -H "Content-Type: application/vnd.git-lfs+json" \
        -d "{\"operation\":\"download\",\"transfers\":[\"basic\"],\"objects\":[{\"oid\":\"$oid\",\"size\":$size}]}" \
        "https://lfs.github.com/$REPO/objects/batch" |
        python3 -c 'import json,sys; print(json.load(sys.stdin)["objects"][0]["actions"]["download"]["href"])')
    curl -sSL -o "$OUT/$m.nc" "$href"
    cp "$POINTERS/$m.json" "$OUT/$m.json"
    [ "$(sha256sum "$OUT/$m.nc" | cut -d' ' -f1)" = "$oid" ] || { echo "sha256 mismatch: $m" >&2; exit 1; }
    echo "fetched $m ($size bytes)"
done
