python -m pip install --no-cache-dir logzero transformers uvicorn fastapi pydantic \
torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
python3 -m convbot_fastapi
