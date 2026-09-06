#!/usr/bin/env bash
# Build one ventus runtime image from its pinned public base.
#
#   ./scripts/build.sh 025          # default 0731 lane
#   ./scripts/build.sh 025-vision   # Vision-Exp lane
#   ./scripts/build.sh 021          # legacy rollback lane
#   VENTUS_TAG=ventus/dspark-vllm:dev ./scripts/build.sh 025
set -euo pipefail
cd "$(dirname "$0")/.."

LANE="${1:-025}"

case "$LANE" in
  025)
    TAG="${VENTUS_TAG:-ghcr.io/ventus-software-solutions/dspark-vllm:0731-025-0.1.0}"
    echo "== 0.25 lane: protocol + NVFP4 dispatch overlay on Anemll GX10"
    docker build -f docker/Dockerfile.overlay-025 -t "$TAG" docker
    ;;

  025-vision)
    TAG="${VENTUS_TAG:-ghcr.io/ventus-software-solutions/dspark-vllm:0731-025-vision-0.1.0}"
    echo "== 0.25 Vision-Exp lane: native image-input overlay"
    docker build -f docker/overlay-025-vision/Dockerfile -t "$TAG" \
      docker/overlay-025-vision
    ;;

  021)
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
    ;;

  *)
    echo "usage: $0 [025|025-vision|021]" >&2
    exit 2
    ;;
esac

echo "built: $TAG"
