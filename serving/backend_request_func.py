import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    model: str
    temperature: int = 0.0
    max_tokens: int = 512
    top_p: int = 1.0


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0
    ttft: float = 0
    prompt_len: int = 0
    prompt: str = ""


async def async_request_vllm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "prompt": request_func_input.prompt,
            "temperature": request_func_input.temperature,
            "top_p": request_func_input.top_p,
            "max_tokens": request_func_input.max_tokens,
            "stream": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.prompt = request_func_input.prompt

        ttft = 0
        st = time.perf_counter()
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                    output.latency = time.perf_counter() - st

                    # When streaming, '\0' is appended to the end of the response.
                    body = data.decode("utf-8").rstrip("\0").split("\0")[-1]
                    try: 
                        output.generated_text = json.loads(
                            body)["text"][0][len(request_func_input.prompt):]
                        output.success = True
                    except Exception as e:
                        with open("debug.jsonl",'a+') as fh:
                            fh.write(body)
                            fh.write("\n")
                        print(e)
                        print(f"{body!r}")
                        print(f"{data!r}")

                else:
                    output.success = False
        except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError):
            output.success = False

        if pbar:
            pbar.update(1)
        return output

ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_vllm,
}
