#!/usr/bin/env bash
# docker/build.sh <tag> [openpi|torch ...]: base plus the runtime images (default both); OPENPI_REF=<sha> overrides the openpi commit.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:?usage: docker/build.sh <tag> [openpi|torch ...]}"
shift
OPENPI_REF="${OPENPI_REF:-215abfb217dbac7d5f1273282331b9b1866c0479}"
RUNTIMES=("$@")
[[ ${#RUNTIMES[@]} -gt 0 ]] || RUNTIMES=(openpi torch)

docker build -f docker/Dockerfile.base -t "umi-arena-base:${TAG}" .
for rt in "${RUNTIMES[@]}"; do
  docker build -f "docker/Dockerfile.${rt}" --build-arg "BASE=umi-arena-base:${TAG}" \
    --build-arg "OPENPI_REF=${OPENPI_REF}" -t "umi-arena-${rt}:${TAG}" .
done
