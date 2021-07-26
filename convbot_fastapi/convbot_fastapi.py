"""Define convbot_fastapi.

params
    max_length: int = 1000,
    do_sample: bool = True,
    top_p: float = 0.95,
    top_k: int = 0,
    temperature: float = 0.75,
"""
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from joblib import Memory
from fastapi import FastAPI
from pydantic import BaseModel

import logzero
from logzero import logger

from .force_async import force_async

logzero.loglevel(10)  # debug

# model_name = "microsoft/DialoGPT-large"
# model_name = "microsoft/DialoGPT-small"
# pylint: disable=invalid-name
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

memory = Memory("cachedir", verbose=False)
app = FastAPI(title="convbot-fastapi")


def convbot_fastapi():
    """Define."""
    logger.debug(" entry ")


class Text(BaseModel):
    text: str
    prev_resp: str
    max_length: int = 1000
    do_sample: bool = True
    top_p: float = 0.95
    top_k: int = 0
    temperature: float = 0.75
    description: Optional[str] = None


@app.post("/text/")
async def post_text(q: Text):
    """Post."""
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
    try:
        _ = await _convbot(
            text, prev_resp, max_length, do_sample, top_p, top_k, temperature,
        )
        _ = {"resp": _}
    except Exception as exc:
        logger.error(exc)
        _ = {"error": True, "message": str(exc)}

    return {"q": q, "result": _}


@memory.cache(verbose=False)
def encode(text):
    """Encode sents."""
    return tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")


@force_async
def _convbot(
    sent: str,
    prev_resp: str = "",
    max_length: int = 1000,
    do_sample: bool = True,
    top_p: float = 0.95,
    top_k: int = 0,
    temperature: float = 0.75,
):
    """Generate a response."""
    # sent_ids = tokenizer.encode(sent + tokenizer.eos_token, return_tensors="pt")
    sent_ids = encode(sent)

    if not prev_resp.strip():
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
