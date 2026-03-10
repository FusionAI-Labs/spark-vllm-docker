import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

import os


os.environ["OPENAI_API_KEY"] = "1234"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8008/v1"


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise ValueError(
            f"Config error: file not found: {config_path}. Provide a valid JSON config with --config."
        )
    try:
        with config_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Config error: invalid JSON in {config_path}: {exc.msg}. Fix the file and retry."
        ) from exc

    if not isinstance(loaded, dict):
        raise ValueError(
            "Config error: top-level JSON must be an object. Include at least model, request, and prompt/messages."
        )

    return loaded


def validate_config(config: dict) -> None:
    if not isinstance(config.get("model"), str) or not config["model"].strip():
        raise ValueError("Config error: 'model' must be a non-empty string.")

    has_prompt = isinstance(config.get("prompt"), str) and bool(
        config["prompt"].strip()
    )
    has_messages = (
        isinstance(config.get("messages"), list) and len(config["messages"]) > 0
    )
    if not has_prompt and not has_messages:
        raise ValueError(
            "Config error: provide either non-empty 'prompt' or non-empty 'messages'."
        )

    request = config.get("request")
    if not isinstance(request, dict):
        raise ValueError("Config error: 'request' must be an object.")
    if "stream" not in request or not isinstance(request["stream"], bool):
        raise ValueError("Config error: 'request.stream' must be set to true or false.")

    if "extra_body" in config and not isinstance(config["extra_body"], dict):
        raise ValueError("Config error: 'extra_body' must be an object when provided.")


def build_messages(config: dict) -> list[ChatCompletionMessageParam]:
    if "messages" in config:
        return config["messages"]
    prompt = config["prompt"]
    return [{"role": "user", "content": prompt}]


def estimate_tokens_per_second(output_text: str, elapsed_s: float) -> tuple[int, float]:
    estimated_tokens = max(1, round(len(output_text) / 4))
    tokens_per_second = estimated_tokens / elapsed_s if elapsed_s > 0 else 0.0
    return estimated_tokens, tokens_per_second


def safe_elapsed(started_at: float | None, ended_at: float) -> float:
    if started_at is None:
        return 0.0
    return max(ended_at - started_at, 1e-9)


def parts_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""


def extract_reasoning_delta(delta_obj: Any) -> str:
    if not delta_obj:
        return ""

    reasoning_content = getattr(delta_obj, "reasoning_content", None)
    text = parts_to_text(reasoning_content)
    if text:
        return text

    reasoning = getattr(delta_obj, "reasoning", None)
    text = parts_to_text(reasoning)
    if text:
        return text

    return ""


def extract_answer_delta(delta_obj: Any) -> str:
    if not delta_obj:
        return ""
    return parts_to_text(getattr(delta_obj, "content", None))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick chat completion test")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file (required)",
    )
    parser.add_argument(
        "--debug-stream",
        action="store_true",
        help="Print raw streamed delta structure at the end",
    )
    args = parser.parse_args()

    if args.config is None:
        print(
            "Config error: missing --config. Run: python3 python/test.py --config python/test.config.json",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        config = load_config(args.config)
        validate_config(config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Expected config fields: model, request.stream, and prompt or messages.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    client = OpenAI()
    messages = build_messages(config)
    request = config["request"]

    create_kwargs = {
        "model": config["model"],
        "messages": messages,
        **request,
    }

    extra_body = config.get("extra_body")
    if extra_body:
        create_kwargs["extra_body"] = extra_body

    print("Assistant:\n")
    started_at = perf_counter()
    answer_started_at: float | None = None
    thinking_text = ""
    answer_text = ""

    if request.get("stream", True):
        stream = client.chat.completions.create(**create_kwargs)
        thinking_chunks: list[str] = []
        answer_chunks: list[str] = []
        printed_thinking_header = False
        printed_answer_header = False
        stream_debug: list[dict[str, Any]] = []
        for event in stream:
            if not event.choices:
                continue
            delta_obj = event.choices[0].delta

            delta_dump = delta_obj.model_dump(exclude_none=True)
            stream_debug.append(
                {
                    "delta_keys": list(delta_dump.keys()),
                    "delta": delta_dump,
                    "finish_reason": event.choices[0].finish_reason,
                }
            )

            reasoning_delta = extract_reasoning_delta(delta_obj)
            if reasoning_delta:
                if not printed_thinking_header:
                    print("[THINKING]\n", end="", flush=True)
                    printed_thinking_header = True
                thinking_chunks.append(reasoning_delta)
                print(reasoning_delta, end="", flush=True)

            answer_delta = extract_answer_delta(delta_obj)
            if answer_delta:
                if not printed_answer_header:
                    if printed_thinking_header:
                        print("\n\n[ANSWER]\n", end="", flush=True)
                    answer_started_at = perf_counter()
                    printed_answer_header = True
                answer_chunks.append(answer_delta)
                print(answer_delta, end="", flush=True)

        thinking_text = "".join(thinking_chunks)
        answer_text = "".join(answer_chunks)

        if args.debug_stream:
            print("\n\n[STREAM DEBUG STRUCTURE]")
            print(json.dumps(stream_debug, indent=2))
    else:
        response = client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message
        thinking_text = parts_to_text(
            getattr(message, "reasoning_content", None)
        ) or parts_to_text(getattr(message, "reasoning", None))
        answer_text = message.content or ""

        if thinking_text:
            print("[THINKING]\n", end="", flush=True)
            print(thinking_text, end="", flush=True)
            if answer_text:
                print("\n\n[ANSWER]\n", end="", flush=True)
        if answer_text:
            print(answer_text, end="", flush=True)

    output_text = f"{thinking_text}{answer_text}"

    ended_at = perf_counter()
    elapsed_s = safe_elapsed(started_at, ended_at)
    estimated_tokens, tokens_per_second = estimate_tokens_per_second(
        output_text, elapsed_s
    )
    print("\n\n----------------------------------------")
    print(
        f"\n\nEstimated throughput: {tokens_per_second:.2f} tokens/s ({estimated_tokens} tokens in {elapsed_s:.2f}s)"
    )

    if thinking_text and answer_text and request.get("stream", True):
        thinking_elapsed_s = (
            safe_elapsed(started_at, answer_started_at)
            if answer_started_at is not None
            else elapsed_s
        )
        answer_elapsed_s = safe_elapsed(answer_started_at, ended_at)

        thinking_tokens, thinking_tps = estimate_tokens_per_second(
            thinking_text, thinking_elapsed_s
        )
        answer_tokens, answer_tps = estimate_tokens_per_second(
            answer_text, answer_elapsed_s
        )
        print(
            f"Thinking throughput: {thinking_tps:.2f} tokens/s ({thinking_tokens} tokens in {thinking_elapsed_s:.2f}s)"
        )
        print(
            f"Answer throughput: {answer_tps:.2f} tokens/s ({answer_tokens} tokens in {answer_elapsed_s:.2f}s)"
        )


if __name__ == "__main__":
    main()
