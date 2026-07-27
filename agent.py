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
    # TODO: gpt-5 계열 정확한 가격이 확정되면 아래 0 값을 실제 가격으로 갱신할 것.
    # (reasoning_tokens도 completion_tokens에 포함되어 과금되므로, usage["reasoning_tokens"]로 별도 확인 가능)
    "gpt-5.4": {"input": 0, "output": 0},
    "gpt-5.6": {"input": 0, "output": 0},
    "gpt-5-mini": {"input": 0, "output": 0},
}

SUPPORTED_MODELS = set(PRICING_USD_PER_1M_TOKENS.keys())

# 400 에러가 나면 문제 파라미터를 하나씩 제거하고 재시도할 후보 목록.
# (gpt-5 계열 추론 모델은 temperature 등을 기본값 외에는 거부하는 경우가 있음 - 실측으로 확인함)
RETRYABLE_PARAM_NAMES = ["temperature", "top_p", "frequency_penalty", "presence_penalty"]

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


def _extract_offending_param(error: "openai.BadRequestError", candidates: list[str]) -> str | None:
    """400 에러에서 문제가 된 파라미터 이름을 찾는다. candidates에 없으면 None."""
    if error.param in candidates:
        return error.param
    message = str(error)
    for name in candidates:
        if name in message:
            return name
    return None


def _call_chat_completions(
    client: OpenAI, model_id: str, messages: list[dict], params: dict, **extra_kwargs: Any
) -> tuple[Any, list[str]]:
    """gpt-5 계열처럼 max_tokens/temperature 등을 지원하지 않는 모델을 위해,
    400 에러가 나면 문제 파라미터를 하나씩 제거하고 재시도한다.
    extra_kwargs는 response_format 등 그대로 전달하고 싶은 추가 인자(재시도 대상 아님).
    반환: (API 응답, 실제로 제거된 파라미터 이름 목록)
    """
    kwargs = {"model": model_id, "messages": messages, **extra_kwargs}
    if model_id.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = params["max_tokens"]
    else:
        kwargs["max_tokens"] = params["max_tokens"]

    for name in RETRYABLE_PARAM_NAMES:
        kwargs[name] = params[name]

    dropped_params: list[str] = []
    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            return response, dropped_params
        except openai.BadRequestError as e:
            offending = _extract_offending_param(e, RETRYABLE_PARAM_NAMES)
            if offending is None or offending not in kwargs:
                raise
            del kwargs[offending]
            dropped_params.append(offending)


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
    model_id: SUPPORTED_MODELS(=PRICING_USD_PER_1M_TOKENS의 키)에 등록된 모델 중 하나.
              "gpt-5"로 시작하는 모델은 max_tokens 대신 max_completion_tokens를 사용하고,
              temperature 등 지원하지 않는 파라미터가 있으면 자동으로 제거 후 재시도한다
              (아래 dropped_params 참고).
    generation_params: temperature/top_p/max_tokens/frequency_penalty/presence_penalty
                        중 일부만 넘겨도 되며, 나머지는 기본값으로 채워진다.
    history: 이전 턴의 {"role": "user"|"assistant", "content": ...} 메시지 목록.
              None(기본값)이면 system + question 단일 턴으로 호출한다(기존 동작과 동일).
              여러 턴짜리 대화를 이어가려면 이전 턴들을 순서대로 넘긴다.
    total_calls: 10회마다 출력되는 누적 비용 로그에 "N/total_calls" 형태로 표시할
                 전체 예상 호출 수. None이면 "N"만 표시한다.

    반환 dict에는 기존 필드에 더해:
    - dropped_params: 400 에러로 인해 자동 제거된 파라미터 이름 목록 (예: ["temperature"]).
                       아무것도 제거되지 않았으면 빈 리스트. 통제 조건이 깨졌는지 보고서에 남길 때 사용.
    - usage["reasoning_tokens"]: 모델이 reasoning 토큰 사용량을 보고하는 경우에만 포함
                                  (completion_tokens에 이미 포함되어 과금되는 값이라, 비용 계산과는 별개로
                                  참고용으로 노출).
    - truncated: 응답이 비어 있거나(reasoning 토큰이 출력 예산을 다 써버린 경우 등)
                 finish_reason이 "length"면 True. gpt-5 계열처럼 reasoning과 응답이
                 같은 max_tokens 예산을 나눠 쓰는 모델에서 특히 중요한 신호라, 발생 시
                 경고를 출력하고 이 필드로도 남긴다.
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
        response, dropped_params = _call_chat_completions(client, model_id, messages, params)
    except openai.AuthenticationError as e:
        raise AgentAPIError("API 키가 유효하지 않습니다. .env의 OPENAI_API_KEY를 확인해주세요.") from e
    except openai.RateLimitError as e:
        raise AgentAPIError("API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.") from e
    except openai.APIConnectionError as e:
        raise AgentAPIError("네트워크 연결에 문제가 발생했습니다. 인터넷 연결을 확인해주세요.") from e
    except openai.APIStatusError as e:
        raise AgentAPIError(f"OpenAI API 오류가 발생했습니다 (status={e.status_code}).") from e

    parsed_response = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    truncated = (not parsed_response) or (finish_reason == "length")
    if truncated:
        print(
            f"[경고] 응답이 잘렸거나 비어 있습니다 (model={model_id}, finish_reason={finish_reason!r}, "
            f"응답 길이={len(parsed_response or '')}). max_tokens을 늘리는 것을 고려하세요."
        )

    _record_cost_and_maybe_print(
        model_id, response.usage.prompt_tokens, response.usage.completion_tokens, total_calls
    )

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    completion_details = getattr(response.usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
    if reasoning_tokens is not None:
        usage["reasoning_tokens"] = reasoning_tokens

    return {
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "persona_id": persona_row.get("uuid"),
        "system_prompt": system_prompt,
        "user_message": question,
        "raw_response": response.model_dump(),
        "parsed_response": parsed_response,
        "usage": usage,
        "dropped_params": dropped_params,
        "truncated": truncated,
    }
