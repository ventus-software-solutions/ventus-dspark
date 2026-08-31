# VISION trial image: our production 025 lane + MiaAI's native Vision-Exp
# support (patches/ fetched from MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
# @ 7963d432, MIT-side community work — see CREDITS.md). Adds the 32-layer ViT
# + Aligner, image_url handling, and the checkpoint's own encoder. Our
# thinking-translation and nvfp4 overlays ride in from the base.
FROM ghcr.io/ventus-software-solutions/dspark-vllm:0731-025-0.1.0

COPY encoding_dsv4.py /opt/ventus/encoding_dsv4.py
COPY patches/hotfix-dsv4-vision-exp.py /opt/hotfix-dsv4-vision-exp.py
COPY patches/hotfix-vllm-empty-encoder-output.py /opt/hotfix-vllm-empty-encoder-output.py
COPY patches/vision_exp /opt/dspark-patches/vision_exp

# Order mirrors their compose: encoder first, then the vision hotfix (it
# patches the encoder it finds), then the empty-encoder-output guard.
RUN python3 -c "import PIL" 2>/dev/null || pip install --no-cache-dir pillow
RUN cp /opt/ventus/encoding_dsv4.py /usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py \
  && python3 /opt/hotfix-dsv4-vision-exp.py \
  && python3 /opt/hotfix-vllm-empty-encoder-output.py \
  && python3 -c "import vllm; print('vllm', vllm.__version__)"
