"""Define convbot_fastapi.

params
    max_length: int = 1000,
    do_sample: bool = True,
    top_p: float = 0.95,
    top_k: int = 0,
    temperature: float = 0.75,
"""
# pylint: disable=invalid-name
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from joblib import Memory
from fastapi import FastAPI, Query
from pydantic import BaseModel

import logzero
from logzero import logger

from .force_async import force_async

logzero.loglevel(10)  # debug

# model_name = "microsoft/DialoGPT-large"
# model_name = "microsoft/DialoGPT-medium"

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

memory = Memory("cachedir", verbose=False)
app = FastAPI(title="convbot-fastapi")


def convbot_fastapi():
    """Define."""
    logger.debug(" entry ")


class Text(BaseModel):
    """Define post format."""

    text: Optional[str]
    prev_resp: Optional[str] = ""
    max_length: int = 1000
    do_sample: bool = True
    top_p: float = 0.95
    top_k: int = 0
    temperature: float = 0.75
    description: Optional[str] = None


@app.get("/")
async def landing(q: str = "") -> str:
    """Feed a landing message."""
    return "convbot-fastapi docs at http://convbot-yucongo.koyeb.app/docs" ""


@app.post("/text/")
async def post_text(q: Text) -> dict:
    r"""Post API for convbot.

    Parameters
        text: str
        prev_resp: str = ""
        max_length: int = 1000
        do_sample: bool = True
        top_p: float = 0.95
        top_k: int = 0
        temperature: float = 0.75
        description: Optional[str] = None

    Returns
        reply: {"q": q, "result": reply} or
            {"q": q, "result": {"error": True, "message": str(exc)}}

    Swagger available or interactive testing at http://.../docs

    Pythn code examples using `requests`:
    ```python
    import requests

    message = "How are you?"

    # url = "api_address"  # e.g. url = "http://convbot.ttw.workers.dev/text"

    url = "http://convbot-yucongo.koyeb.app/text"
    url = "http://acone3:8000"
    data = {"text": message, "prev_resp": ""}
    res = requests.post(f"{url}/text", json=data)
    reply = res.json().get("result").get("resp")
    print(reply)
    # 'Good, you?' or something

    # or with relpy history context
    prev_resp = "I am fine."
    message1 = "What's your name?"
    data = {"text": message1, "prev_resp": prev_resp}
    res = requests.post(f"{url}/text", json=data)
    reply1 = res.json().get("result").get("resp")

    # wrap reuests.post in `try:... except...` as desired
    # in case there are network problems

    # Other parameters (`max_length`, `do_sample`...etc.) can \also be supplied.
    ```

    Replies from the interface are not deterministic, i.e.,
    if you feed the same `text` and `prev_resp` twice, the returned
    replies will likely not be the same.

    `convbot` occasionally returns empty replies. In these cases, you
    may wish to change the history context (`prev_resp`) or set
    prev_resp as "" (empty) and try again.

    An interactive loop might look like this:
    ```python
    import requests
    url = "..."  # plug in a valid address
    url = "http://convbot-yucongo.koyeb.app"
    url = "http://acone3:8000"

    prev_resp = ""
    print("Bot: talk to me (to exit, type quit)")
    while 1:
        user_msg = input("User: ")
        if user_msg.lower() in ["quit", "exit"]:
            print("bye...")
            break
        r = requests.post(f"{url}/text", json={"text": user_msg, "prev_resp": prev_resp})
        resp = r.json().get("result").get("result")
        print("Bot: ", resp)
        prev_resp = resp
    """
    text = q.text
    prev_resp = q.prev_resp
    max_length = q.max_length
    do_sample = q.do_sample
    top_p = q.top_p
    top_k = q.top_k
    temperature = q.temperature

    logger.debug("text: %s", text)

    # _ = sent_corr(text1, text2)
    # _ = await deepl_tr(text, from_lang, to_lang, page=PAGE,)
    if text is None:
        text = ""
    if prev_resp is None:
        prev_resp = ""
    try:
        _ = await _convbot(
            text, prev_resp, max_length, do_sample, top_p, top_k, temperature,
        )
        _ = {"resp": _}
    except Exception as exc:
        logger.error(exc)
        _ = {"resp": "", "error": True, "message": str(exc)}

    return {"q": q, "result": _}


@app.get("/text/")
async def get_text(
    # https://fastapi.tiangolo.com/tutorial/query-params/
    msg: Optional[str] = Query(
        ...,
        max_length=1500,
        min_length=2,  # disallow one character message
        title="user's message",
        description="max. 5000 chars.",
    ),
    prev_resp: Optional[str] = Query(
        "",  # default empty str
        max_length=1500,
        title="bot's previous reply (history)",
        description="max. 5000 chars.",
    ),
    max_length: int = 1000,
    do_sample: bool = True,
    top_p: float = 0.95,
    top_k: int = 0,
    temperature: float = 0.75,
    description: Optional[str] = None,
) -> dict:
    """Get API for convbot.

    Python example
    ```python
    import requests

    msg = "How are you?"
    url = "http://convbot-yucongo.koyeb.app"
    url = "http://acone3:8000"
    res = requests.get(f"{url}/text/?msg={msg}")
    reply = res.json().get("result").get("resp")
    print(reply)
    # 'I am good' or the like
    ```
    """
    logger.debug("text: %s", msg)
    # prepareing return dict
    q = dict(
        msg=msg,
        prev_resp=prev_resp,
        max_length=max_length,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        description=description,
    )

    if msg:
        text = msg
    else:
        text = ""

    if not prev_resp:  # Taking care of None
        prev_resp = ""

    # _ = sent_corr(text1, text2)
    # _ = await deepl_tr(text, from_lang, to_lang, page=PAGE,)
    try:
        _ = await _convbot(
            text, prev_resp, max_length, do_sample, top_p, top_k, temperature,
        )
        _ = {"resp": _}
    except Exception as exc:
        logger.error(exc)
        _ = {"error": True, "message": str(exc)}

    logger.debug("return: %s", q)

    return {"q": q, "result": _}


@memory.cache(verbose=False)
def encode(text):
    """Encode sents."""
    return tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")


@force_async
def _convbot(
    sent: Optional[str],
    prev_resp: Optional[str] = "",
    max_length: int = 1000,
    do_sample: bool = True,
    top_p: float = 0.95,
    top_k: int = 0,
    temperature: float = 0.75,
):
    """Generate a response."""
    # sent_ids = tokenizer.encode(sent + tokenizer.eos_token, return_tensors="pt")
    if sent is None:
        sent = ""
    sent_ids = encode(sent)

    if not prev_resp or not prev_resp.strip():
        # prev_resp_ids = tokenizer.encode(prev_resp + tokenizer.eos_token, return_tensors="pt")
        prev_resp_ids = encode(prev_resp)
        bot_input_ids = torch.cat([prev_resp_ids, sent_ids], dim=-1)
    else:
        bot_input_ids = sent_ids

    output = model.generate(
        bot_input_ids,
        max_length=max_length,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    resp = tokenizer.decode(
        output[:, bot_input_ids.shape[-1] :][0], skip_special_tokens=True
    )

    return resp
