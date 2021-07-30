# convbot-fastapi
[![tests](https://github.com/ffreemt/convbot-fastapi/actions/workflows/routine-tests.yml/badge.svg)](https://github.com/ffreemt/convbot_fastapi/actions)[![python](https://img.shields.io/static/v1?label=python+&message=3.7%2B&color=blue)](https://img.shields.io/static/v1?label=python+&message=3.7%2B&color=blue)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![PyPI version](https://badge.fury.io/py/convbot_fastapi.svg)](https://badge.fury.io/py/convbot_fastapi)

A conversational chatbot API via `fastapi`

## Install it

```shell
pip install convbot-fastapi
# or poetry add convbot-fastapi
# pip install git+htts://github.com/ffreemt/convbot-fastapi
# poetry add git+htts://github.com/ffreemt/convbot-fastapi

# To upgrade
pip install convbot-fastapi -U
# or poetry add convbot-fastapi@latest
```

## Use it
```bash
python -m convbot_fastapi
```
Point your browser to `httpL//127.0.0.1:8000/docs`

Or with uvicorn
```bash
uvicorn convbot_fastapi.convbot_fastapi:app
```

## Deploy it to the cloud
Clone the repo
```bash
git clone https://github.com/ffreemt/convbot-fastapi
cd convbot-fastapi
```

Create a docker image from `Dockerfile` and upload it to `https://hub.docker.com/`

Sign up and deploy the docker image to, for example, `koyeb.com`.
