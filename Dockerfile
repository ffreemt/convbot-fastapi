FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8 as runner

WORKDIR /convbot-fatapi

# RUN python3 -m pip install poetry && poetry config virtualenvs.create false

COPY . ./

# RUN poetry install --no-root --no-dev

# RUN python -m pip install --no-cache-dir logzero transformers uvicorn fastapi pydantic \
# torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

EXPOSE 8000

# CMD python3 -m convbot_fastapi
CMD bash install-and-run.sh
