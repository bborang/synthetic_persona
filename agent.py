"""H3 실험용 페르소나 질의 모듈.

main.py의 system prompt 생성(build_system_prompt)과 API 키 로딩(load_api_key)
로직을 재사용해서, 지정한 모델로 페르소나에게 질문하고 구조화된 결과를 반환한다.
"""

from datetime import datetime, timezone
from typing import Any

import openai
from openai import OpenAI

import pandas as pd

from main import build_system_prompt, load_api_key

# 가격이 바뀌면 이 dict만 수정하면 된다 (단위: USD / 1M tokens).
PRICING_USD_PER_1M_TOKENS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}

SUPPORTED_MODELS = set(PRICING_USD_PER_1M_TOKENS.keys())

DEFAULT_GENERATION_PARAMS = {
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 1024,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

_client: OpenAI | None = None

_cost_tracker = {
    "call_count": 0,
    "total_input_cost": 0.0,
    "total_output_cost": 0.0,
}


def _record_cost_and_maybe_print(model_id: str, prompt_tokens: int, completion_tokens: int, total_calls: int | None) -> None:
    """호출 비용을 누적하고, 10회마다 지금까지의 누적 비용을 출력."""
    pricing = PRICING_USD_PER_1M_TOKENS[model_id]
    input_cost = prompt_tokens / 1_000_000 * pricing["input"]
    output_cost = completion_tokens / 1_000_000 * pricing["output"]

    _cost_tracker["call_count"] += 1
    _cost_tracker["total_input_cost"] += input_cost
    _cost_tracker["total_output_cost"] += output_cost

    if _cost_tracker["call_count"] % 10 == 0:
        total_cost = _cost_tracker["total_input_cost"] + _cost_tracker["total_output_cost"]
        call_count = _cost_tracker["call_count"]
        call_label = f"{call_count}/{total_calls}" if total_calls else str(call_count)
        print(
            f"[호출 {call_label}] 누적 비용: ${total_cost:.2f} "
            f"(입력: ${_cost_tracker['total_input_cost']:.2f}, 출력: ${_cost_tracker['total_output_cost']:.2f})"
        )


class AgentAPIError(RuntimeError):
    """OpenAI API 호출 실패를 나타내는 예외."""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=load_api_key())
    return _client


def ask_persona(
    persona_row: pd.Series,
    question: str,
    model_id: str = "gpt-4o-mini",
    generation_params: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
    total_calls: int | None = None,
) -> dict[str, Any]:
    """페르소나 1명(persona_row)에게 question을 물어보고 구조화된 결과를 반환.

    persona_row: main.py의 페르소나 스키마(uuid 포함 51개 컬럼)를 따르는 pd.Series
    model_id: "gpt-4o" | "gpt-4o-mini" | "gpt-4.1"
    generation_params: temperature/top_p/max_tokens/frequency_penalty/presence_penalty
                        중 일부만 넘겨도 되며, 나머지는 기본값으로 채워진다.
    history: 이전 턴의 {"role": "user"|"assistant", "content": ...} 메시지 목록.
              None(기본값)이면 system + question 단일 턴으로 호출한다(기존 동작과 동일).
              여러 턴짜리 대화를 이어가려면 이전 턴들을 순서대로 넘긴다.
    total_calls: 10회마다 출력되는 누적 비용 로그에 "N/total_calls" 형태로 표시할
                 전체 예상 호출 수. None이면 "N"만 표시한다.
    """
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"지원하지 않는 model_id입니다: {model_id!r} (지원 모델: {sorted(SUPPORTED_MODELS)})")

    params = {**DEFAULT_GENERATION_PARAMS, **(generation_params or {})}
    system_prompt = build_system_prompt(persona_row)
    client = _get_client()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            frequency_penalty=params["frequency_penalty"],
            presence_penalty=params["presence_penalty"],
        )
    except openai.AuthenticationError as e:
        raise AgentAPIError("API 키가 유효하지 않습니다. .env의 OPENAI_API_KEY를 확인해주세요.") from e
    except openai.RateLimitError as e:
        raise AgentAPIError("API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.") from e
    except openai.APIConnectionError as e:
        raise AgentAPIError("네트워크 연결에 문제가 발생했습니다. 인터넷 연결을 확인해주세요.") from e
    except openai.APIStatusError as e:
        raise AgentAPIError(f"OpenAI API 오류가 발생했습니다 (status={e.status_code}).") from e

    parsed_response = response.choices[0].message.content

    _record_cost_and_maybe_print(
        model_id, response.usage.prompt_tokens, response.usage.completion_tokens, total_calls
    )

    return {
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "persona_id": persona_row.get("uuid"),
        "system_prompt": system_prompt,
        "user_message": question,
        "raw_response": response.model_dump(),
        "parsed_response": parsed_response,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }
