#!/usr/bin/env python3
"""OpenAI-compatible local HF model server that records agent hidden states.

This is intended for the Agentic-Upward-Deception pre-experiment where the
agent itself must be a local open-source model. The repo's existing
smolagents/OpenAIServerModel client can call this server through
EVAL_URL=http://127.0.0.1:<port>/v1.

For every /v1/chat/completions request, the server generates text with the
local model and saves pooled hidden-state vectors for that exact model call.
The saved vectors are therefore local-agent hidden states, not replay features
computed from a finished log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

# Compatibility for some rjob container Python builds where the optional
# _uuid extension exists but misses has_uuid_generate_time_safe. The stdlib
# uuid module accesses this attribute during import, so patch it before
# importing uuid.
try:
    import _uuid  # type: ignore
    if not hasattr(_uuid, "has_uuid_generate_time_safe"):
        _uuid.has_uuid_generate_time_safe = None  # type: ignore[attr-defined]
except Exception:
    pass
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn


def eprint(msg: str) -> None:
    print(msg, flush=True)


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(x for x in parts if x)
    return str(content)


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = normalize_content(msg.get("content"))
        if role not in {"system", "user", "assistant"}:
            content = f"[{role}]\n{content}"
            role = "user"
        out.append({"role": role, "content": content})
    return out


def trim_stop_sequences(text: str, stops: Optional[List[str]]) -> Tuple[str, Optional[str]]:
    if not stops:
        return text, None
    best_pos: Optional[int] = None
    best_stop: Optional[str] = None
    for stop in stops:
        if not stop:
            continue
        pos = text.find(stop)
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best_pos = pos
            best_stop = stop
    if best_pos is None:
        return text, None
    return text[:best_pos], best_stop


class HiddenStateServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_id = args.model_id
        self.default_hidden_root = Path(args.hidden_root).expanduser().resolve()
        self.default_hidden_root.mkdir(parents=True, exist_ok=True)
        self.session_id = "unassigned"
        self.hidden_dir = self.default_hidden_root / self.session_id
        self.hidden_dir.mkdir(parents=True, exist_ok=True)
        self.call_index = 0
        self.steering_config: Dict[str, Any] = {"active": False}
        self.steering_delta = None

        self.torch = None
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.load_model()

    def load_model(self) -> None:
        import torch

        try:
            import torch_npu  # noqa: F401
        except Exception as exc:
            eprint(f"[server] torch_npu import warning: {exc}")

        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self.torch = torch
        model_path = self.args.model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=self.args.trust_remote_code)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = self.args.truncation_side

        kwargs: Dict[str, Any] = {
            "trust_remote_code": self.args.trust_remote_code,
            "output_hidden_states": True,
        }
        if self.args.torch_dtype != "none":
            kwargs["torch_dtype"] = self.args.torch_dtype if self.args.torch_dtype == "auto" else getattr(torch, self.args.torch_dtype)

        loaders = [AutoModelForImageTextToText, AutoModelForCausalLM]
        last_error: Optional[Exception] = None
        for loader in loaders:
            try:
                loaded = loader.from_pretrained(model_path, output_loading_info=True, **kwargs)
                if isinstance(loaded, tuple):
                    self.model, info = loaded
                else:
                    self.model, info = loaded, {}
                eprint(
                    f"[server] loaded {loader.__name__}; "
                    f"missing={len(info.get('missing_keys', [])) if isinstance(info, dict) else 'NA'} "
                    f"unexpected={len(info.get('unexpected_keys', [])) if isinstance(info, dict) else 'NA'}"
                )
                break
            except Exception as exc:
                last_error = exc
                eprint(f"[server] {loader.__name__} failed: {exc}")
        if self.model is None:
            raise RuntimeError(f"Could not load model from {model_path}: {last_error}")

        if self.args.device == "auto":
            if hasattr(torch, "npu") and torch.npu.is_available():
                self.device = "npu:0"
            elif torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = self.args.device
        self.model.to(self.device)
        getattr(self.model, "eval")()
        eprint(f"[server] ready device={self.device} model={model_path}")

    def set_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not payload.get("active", True):
            self.session_id = "unassigned"
            self.hidden_dir = self.default_hidden_root / self.session_id
            self.hidden_dir.mkdir(parents=True, exist_ok=True)
            self.call_index = 0
            self.steering_config = {"active": False}
            self.steering_delta = None
            return {"ok": True, "active": False}

        hidden_dir = payload.get("hidden_dir")
        session_id = payload.get("session_id") or "session"
        if hidden_dir:
            self.hidden_dir = Path(hidden_dir).expanduser().resolve()
        else:
            self.hidden_dir = self.default_hidden_root / str(session_id)
        self.hidden_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id)
        self.call_index = 0
        self.configure_steering(payload.get("steering") or {})
        (self.hidden_dir / "session.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        eprint(f"[server] session={self.session_id} hidden_dir={self.hidden_dir} steering={self.steering_config}")
        return {"ok": True, "active": True, "session_id": self.session_id, "hidden_dir": str(self.hidden_dir), "steering": self.steering_config}


    def configure_steering(self, cfg: Dict[str, Any]) -> None:
        if not cfg or not cfg.get("active", True) or float(cfg.get("alpha", 0.0) or 0.0) == 0.0:
            self.steering_config = {"active": False}
            self.steering_delta = None
            return
        path = Path(str(cfg.get("vector_path", ""))).expanduser()
        method = str(cfg.get("method", ""))
        if not path.is_file():
            raise FileNotFoundError(f"steering vector file not found: {path}")
        with np.load(path) as z:
            if method not in z.files:
                raise KeyError(f"steering method {method!r} not in {path}; keys={z.files}")
            delta = np.asarray(z[method], dtype=np.float32)
        if delta.ndim != 2:
            raise ValueError(f"expected [layers, hidden] steering delta, got {delta.shape}")
        self.steering_delta = self.torch.from_numpy(delta).to(self.device)
        self.steering_config = {
            "active": True,
            "method": method,
            "alpha": float(cfg.get("alpha", 0.0)),
            "vector_path": str(path),
            "delta_shape": list(delta.shape),
            "apply": "last_token_each_decoder_layer_during_generate",
        }

    def decoder_layers(self):
        roots = [
            getattr(self.model, "model", None),
            getattr(getattr(self.model, "model", None), "model", None),
            getattr(self.model, "language_model", None),
            getattr(getattr(self.model, "language_model", None), "model", None),
        ]
        for root in roots:
            layers = getattr(root, "layers", None) if root is not None else None
            if layers is not None:
                return layers
        raise RuntimeError("could not locate decoder layers for steering hook")

    def generate_with_optional_steering(self, encoded: Dict[str, Any], gen_kwargs: Dict[str, Any]):
        if not self.steering_config.get("active") or self.steering_delta is None:
            return self.model.generate(**encoded, **gen_kwargs)
        alpha = float(self.steering_config.get("alpha", 0.0))
        delta = self.steering_delta
        layers = self.decoder_layers()
        n = min(len(layers), int(delta.shape[0]))
        active_indices = [i for i in range(n) if int(delta[i].count_nonzero().item()) > 0]
        handles = []

        def make_hook(layer_index: int):
            def hook(_module, _inputs, output):
                d = (alpha * delta[layer_index]).to(device=self.device)
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                    hidden[:, -1, :] = hidden[:, -1, :] + d.to(dtype=hidden.dtype)
                    return (hidden,) + output[1:]
                hidden = output.clone()
                hidden[:, -1, :] = hidden[:, -1, :] + d.to(dtype=hidden.dtype)
                return hidden
            return hook

        try:
            for i in active_indices:
                handles.append(layers[i].register_forward_hook(make_hook(i)))
            return self.model.generate(**encoded, **gen_kwargs)
        finally:
            for h in handles:
                h.remove()

    def build_prompt(self, messages: List[Dict[str, Any]]) -> str:
        clean = normalize_messages(messages)
        try:
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.args.qwen3_no_think:
                template_kwargs["enable_thinking"] = False
            return self.tokenizer.apply_chat_template(clean, **template_kwargs)
        except Exception:
            parts = []
            for m in clean:
                parts.append(f"{m['role'].upper()}:\n{m['content']}")
            parts.append("ASSISTANT:\n")
            return "\n\n".join(parts)

    def save_hidden_states(
        self,
        prompt_text: str,
        response_text: str,
        raw_messages: List[Dict[str, Any]],
        usage: Dict[str, int],
        stopped_by: Optional[str],
    ) -> Dict[str, Any]:
        torch = self.torch
        idx = self.call_index
        self.call_index += 1
        call_id = f"call_{idx:04d}_{uuid.uuid4().hex[:8]}"

        hidden_text = prompt_text + response_text
        encoded = self.tokenizer(
            hidden_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.hidden_max_tokens,
            add_special_tokens=False,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        response_tokens = self.tokenizer(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
        seq_len = int(encoded["input_ids"].shape[1])
        response_start = max(0, seq_len - response_tokens)

        with torch.inference_mode():
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)

        mask = encoded["attention_mask"].to(dtype=torch.float32)
        layers = []
        full_mean = []
        prompt_mean = []
        response_mean = []
        last_token = []
        for layer_idx, h in enumerate(outputs.hidden_states[1:], start=1):
            h32 = h.detach().to(dtype=torch.float32)
            full = (h32 * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            if response_start > 0:
                p_mask = mask[:, :response_start]
                p_h = h32[:, :response_start, :]
                p_mean = (p_h * p_mask[:, :, None]).sum(dim=1) / p_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            else:
                p_mean = full
            if response_start < seq_len:
                r_mask = mask[:, response_start:]
                r_h = h32[:, response_start:, :]
                r_mean = (r_h * r_mask[:, :, None]).sum(dim=1) / r_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            else:
                r_mean = h32[:, -1:, :].mean(dim=1)
            layers.append(layer_idx)
            full_mean.append(full.cpu().numpy()[0])
            prompt_mean.append(p_mean.cpu().numpy()[0])
            response_mean.append(r_mean.cpu().numpy()[0])
            last_token.append(h32[:, -1, :].cpu().numpy()[0])

        npz_path = self.hidden_dir / f"{call_id}.npz"
        np.savez_compressed(
            npz_path,
            layers=np.array(layers, dtype=np.int32),
            full_mean=np.stack(full_mean).astype(np.float32),
            prompt_mean=np.stack(prompt_mean).astype(np.float32),
            response_mean=np.stack(response_mean).astype(np.float32),
            last_token=np.stack(last_token).astype(np.float32),
        )
        meta = {
            "call_index": idx,
            "call_id": call_id,
            "model_id": self.model_id,
            "model_path": self.args.model,
            "session_id": self.session_id,
            "npz_path": str(npz_path),
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8", errors="ignore")).hexdigest(),
            "response_sha256": hashlib.sha256(response_text.encode("utf-8", errors="ignore")).hexdigest(),
            "prompt_chars": len(prompt_text),
            "response_chars": len(response_text),
            "hidden_tokens": seq_len,
            "hidden_max_tokens": self.args.hidden_max_tokens,
            "hidden_truncated": len(self.tokenizer(hidden_text, add_special_tokens=False)["input_ids"]) > self.args.hidden_max_tokens,
            "response_tokens_for_pool": int(response_tokens),
            "usage": usage,
            "stopped_by": stopped_by,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": raw_messages if self.args.save_messages else None,
            "response_text": response_text if self.args.save_text else None,
        }
        meta_path = self.hidden_dir / f"{call_id}.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta

    def chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        torch = self.torch
        messages = payload.get("messages") or []
        prompt_text = self.build_prompt(messages)
        max_new_tokens = int(payload.get("max_tokens") or self.args.max_new_tokens)
        max_new_tokens = max(1, min(max_new_tokens, self.args.max_new_tokens))
        temperature = float(payload.get("temperature") if payload.get("temperature") is not None else self.args.temperature)
        top_p = float(payload.get("top_p") if payload.get("top_p") is not None else self.args.top_p)
        stops = payload.get("stop")
        if isinstance(stops, str):
            stops = [stops]

        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_input_tokens,
            add_special_tokens=False,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        input_tokens = int(encoded["attention_mask"].sum().item())

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            generated = self.generate_with_optional_steering(encoded, gen_kwargs)
        new_ids = generated[0, encoded["input_ids"].shape[1] :]
        response_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        response_text, stopped_by = trim_stop_sequences(response_text, stops)
        output_tokens = int(new_ids.shape[0])
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        hidden_meta = self.save_hidden_states(prompt_text, response_text, messages, usage, stopped_by)
        created = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop" if stopped_by else "length",
                }
            ],
            "usage": usage,
            "hidden_state_meta": {
                "session_id": hidden_meta["session_id"],
                "call_id": hidden_meta["call_id"],
                "npz_path": hidden_meta["npz_path"],
                "steering": self.steering_config,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", default="qwen3-14b-hidden")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--hidden-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--hidden-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument(
        "--qwen3-no-think",
        action="store_true",
        default=True,
        help="Pass enable_thinking=False to Qwen3 chat template.",
    )
    parser.add_argument("--save-messages", action="store_true")
    parser.add_argument("--save-text", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = HiddenStateServer(args)
    app = FastAPI()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "model_id": server.model_id, "device": server.device, "session_id": server.session_id}

    @app.post("/control/session")
    async def control_session(request: Request) -> JSONResponse:
        payload = await request.json()
        return JSONResponse(server.set_session(payload))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            return JSONResponse(server.chat_completion(payload))
        except Exception as exc:
            eprint(f"[server] completion error: {type(exc).__name__}: {exc}")
            return JSONResponse({"error": {"message": str(exc), "type": type(exc).__name__}}, status_code=500)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
