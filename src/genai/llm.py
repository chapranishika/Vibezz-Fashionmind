"""
OpenRouter chat client (OpenAI-compatible) with tool calling.

One HTTPS endpoint, one key, many models — https://openrouter.ai/api/v1.
Used by the stylist when OPENROUTER_API_KEY is set; the Gemini path in
stylist_chatbot.py stays as a fallback.

    run_chat(system, history, user_msg, tools, tool_runner) -> (text, updated_history)

`tools` is the FashionMind TOOL_DECLARATIONS list ({name, description,
parameters}); it is wrapped into OpenAI function-tool shape here.
`tool_runner(name, args_dict) -> str` executes a tool and returns a JSON string.
"""
from __future__ import annotations

import json
import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
_MAX_ROUNDS = 4


def _to_openai_tools(decls: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": d["name"], "description": d.get("description", ""),
        "parameters": d.get("parameters", {"type": "object", "properties": {}}),
    }} for d in decls]


def _hist_to_messages(history: list[dict]) -> list[dict]:
    out = []
    for t in history or []:
        role = t.get("role", "user")
        out.append({"role": "assistant" if role in ("model", "assistant") else "user",
                    "content": t.get("content", "")})
    return out


def _post(payload: dict, api_key: str) -> dict:
    r = requests.post(OPENROUTER_URL, timeout=60, headers={
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://vibezz-fashionmind.vercel.app"),
        "X-Title": "FashionMind Stylist",
        "Content-Type": "application/json",
    }, json=payload)
    r.raise_for_status()
    return r.json()


def run_chat(system: str, history: list[dict], user_msg: str,
             tools: list[dict], tool_runner, model: str | None = None,
             api_key: str | None = None, on_tool=None):
    """Full tool-calling loop. Returns (final_text, updated_history).
    `on_tool(name)` is an optional callback fired when a tool is invoked
    (used by the streaming endpoint to surface tool activity)."""
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    model = model or DEFAULT_MODEL
    oa_tools = _to_openai_tools(tools)

    messages = [{"role": "system", "content": system}]
    messages += _hist_to_messages(history)
    messages.append({"role": "user", "content": user_msg})

    for _ in range(_MAX_ROUNDS):
        data = _post({"model": model, "messages": messages, "tools": oa_tools,
                      "tool_choice": "auto", "temperature": 0.7,
                      "max_tokens": 1024}, key)
        msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            text = (msg.get("content") or "").strip() or \
                   "I can help you find the perfect outfit — what's the occasion?"
            new_hist = list(history or [])
            new_hist.append({"role": "user", "content": user_msg})
            new_hist.append({"role": "model", "content": text})
            return text, new_hist

        # execute each requested tool, feed results back
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            result = tool_runner(name, args)
            if on_tool:
                try:
                    on_tool(name, result)      # newer callers want the result too
                except TypeError:
                    on_tool(name)              # back-compat: name only
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                             "name": name, "content": result})

    # ran out of rounds
    text = "Here are some options based on what's trending — want me to build a full look?"
    new_hist = list(history or [])
    new_hist.append({"role": "user", "content": user_msg})
    new_hist.append({"role": "model", "content": text})
    return text, new_hist
