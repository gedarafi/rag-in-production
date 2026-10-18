# The recommender worker Lambda (src/recommender/worker.py) as a container image.
# Built and pushed to ECR by infra/scripts/build_recommender_image.sh before each plan and apply of the
# recommender unit. The build context is the repository root; .dockerignore keeps it to what is copied here.
FROM public.ecr.aws/lambda/python:3.14

# Runtime dependencies only (no PyTorch): the "recommender" group of pyproject.toml, pinned by uv.lock.
COPY infra/.build/recommender/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --only-binary :all: -r /tmp/requirements.txt && rm /tmp/requirements.txt

# MiniLM in ONNX, pinned by revision (src/recommender/embedding.py), baked in so a cold start downloads nothing.
ENV EMBEDDING_MODEL_DIR=/opt/models/all-MiniLM-L6-v2
COPY src/recommender/embedding.py /tmp/embedding.py
RUN python -c "import importlib.util, pathlib; \
spec = importlib.util.spec_from_file_location('embedding', '/tmp/embedding.py'); \
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); \
module.download_model(pathlib.Path('$EMBEDDING_MODEL_DIR'))" && rm /tmp/embedding.py

COPY src/__init__.py src/tracing.py ${LAMBDA_TASK_ROOT}/src/
COPY src/chat/__init__.py src/chat/store.py ${LAMBDA_TASK_ROOT}/src/chat/
COPY src/recommender/ ${LAMBDA_TASK_ROOT}/src/recommender/

CMD ["src.recommender.worker.handler"]
