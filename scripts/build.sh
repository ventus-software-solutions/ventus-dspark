#!/usr/bin/env bash
# Build the ventus runtime image from pinned public sources.
# Pure Python overlay — no compiled code of our own (see docker/ for the
# exact chain). Run on each DGX Spark node, or in CI on an ARM64 runner.
#
#   ./scripts/build.sh                 # tags ventus/dspark-vllm:0731-0.1.0
#   VENTUS_TAG=ventus/dspark-vllm:dev ./scripts/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_IMAGE="${VENTUS_BASE_IMAGE:-ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready}"
TAG="${VENTUS_TAG:-ghcr.io/ventus-software-solutions/dspark-vllm:0731-0.1.0}"
OVERLAY=ventus/dspark-vllm:overlay
A=ventus/dspark-vllm:nvfp4-a
B=ventus/dspark-vllm:nvfp4-b

echo "== stage 1/4: overlay on $BASE_IMAGE"
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -f docker/Dockerfile.overlay -t "$OVERLAY" docker/overlay
echo "== stage 2/4: nvfp4-a"
docker build --build-arg BASE_IMAGE="$OVERLAY" -f docker/Dockerfile.stage-a -t "$A" .
echo "== stage 3/4: nvfp4-b"
docker build --build-arg BASE_IMAGE="$A" -f docker/Dockerfile.stage-b -t "$B" .
echo "== stage 4/4: nvfp4-stage-c -> $TAG"
docker build --build-arg BASE_IMAGE="$B" -f docker/Dockerfile.stage-c -t "$TAG" .

docker run --rm --entrypoint /opt/env/bin/python "$TAG" -c \
  "import vllm, vllm.v1.spec_decode.dspark as d; print('ventus-dspark image ok', vllm.__version__)"
echo "built: $TAG"
