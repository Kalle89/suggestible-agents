"""Minimal toolkit: load a chat model, render chat/tool prompts per model family, read next-token
distributions over a fixed set of tokens, extract residual-stream directions and add them while the model runs.

Everything is measured as next-token probabilities (no sampling, no LLM judge), so results are deterministic
for a given model, dtype and batch size.
"""
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MODELS = {
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}


def load_json(name):
    return json.loads((DATA / name).read_text())


def load_jsonl(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line.strip()]


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(obj)); tmp.replace(path)


class Model:
    """A Hugging Face chat model plus the family-specific details needed for tool-call prompts."""

    def __init__(self, name, bs=8, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf = MODELS.get(name, name)
        self.name, self.bs = name, bs
        self.family = "qwen3" if "qwen3" in hf.lower() else "llama" if "llama" in hf.lower() else "qwen"
        self.tok = AutoTokenizer.from_pretrained(hf)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=dtype, device_map="auto").eval()
        self.n_layers = self.model.config.num_hidden_layers
        # tool-call formats: the assistant turn is prefilled up to the function name, so the next token picks the tool
        self.argkey = "parameters" if self.family == "llama" else "arguments"
        self.prefill = '{"name": "' if self.family == "llama" else '<tool_call>\n{"name": "'

    def layers(self):
        return self.model.model.layers

    def chat(self, messages, tools=None):
        kw = dict(tokenize=False, add_generation_prompt=True)
        if tools:
            kw["tools"] = tools
        if self.family == "qwen3":
            kw["enable_thinking"] = False
        s = self.tok.apply_chat_template(messages, **kw)
        bos = self.tok.bos_token or ""
        return s[len(bos):] if bos and s.startswith(bos) else s  # the tokenizer adds BOS itself

    def first_token(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"][0]

    def encode(self, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=True)
        mask = enc["attention_mask"]
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        dev = self.model.get_input_embeddings().weight.device
        return enc["input_ids"].to(dev), mask.to(dev), pos.to(dev)

    @contextmanager
    def steer(self, adds=None):
        """adds: {layer: vector}. The vector is added to the residual stream after that decoder layer, at every position."""
        handles = []
        for layer, v in (adds or {}).items():
            def hook(module, inp, out, v=v.float()):
                h = out[0] if isinstance(out, tuple) else out
                h = h + v.to(h.device, h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            handles.append(self.layers()[layer].register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def next_token_probs(self, prompts, ids, adds=None):
        """Probability distribution over the candidate tokens `ids` at the end of each prompt (renormalised over `ids`)."""
        out = []
        with self.steer(adds):
            for i in range(0, len(prompts), self.bs):
                x, m, p = self.encode(prompts[i:i + self.bs])
                logits = self.model(input_ids=x, attention_mask=m, position_ids=p, logits_to_keep=1).logits[:, -1].float()
                out += torch.softmax(logits[:, ids], -1).cpu().tolist()
        return np.asarray(out)

    @torch.no_grad()
    def activations(self, texts):
        """Residual stream after every layer at the last token → tensor [N, n_layers, d] (float32, CPU)."""
        out = []
        for i in range(0, len(texts), self.bs):
            x, m, p = self.encode(texts[i:i + self.bs])
            hs = self.model(input_ids=x, attention_mask=m, position_ids=p, output_hidden_states=True, logits_to_keep=1).hidden_states
            out.append(torch.stack([h[:, -1].float().cpu() for h in hs[1:]], dim=1))
        return torch.cat(out)

    @torch.no_grad()
    def cont_logprob(self, prompts, conts, adds=None):
        """Mean per-token log-probability of conts[i] after prompts[i]."""
        res = []
        with self.steer(adds):
            for i in range(0, len(prompts), self.bs):
                P, C = prompts[i:i + self.bs], conts[i:i + self.bs]
                p_ids = [self.tok(s, add_special_tokens=True)["input_ids"] for s in P]
                c_ids = [self.tok(s, add_special_tokens=False)["input_ids"] for s in C]
                seqs = [a + b for a, b in zip(p_ids, c_ids)]
                L = max(map(len, seqs)); pad = self.tok.pad_token_id
                x = torch.tensor([[pad] * (L - len(s)) + s for s in seqs])
                m = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs])
                pos = (m.cumsum(-1) - 1).clamp(min=0)
                dev = self.model.get_input_embeddings().weight.device
                K = max(map(len, c_ids)) + 1
                lp = F.log_softmax(self.model(input_ids=x.to(dev), attention_mask=m.to(dev), position_ids=pos.to(dev),
                                              logits_to_keep=K).logits.float(), -1).cpu()
                for j, c in enumerate(c_ids):
                    rows = lp[j, K - len(c) - 1:K - 1]
                    res.append(float(rows.gather(-1, torch.tensor(c).unsqueeze(-1)).sum()) / len(c))
        return np.asarray(res)


# ---------------------------------------------------------------- tool helpers

def tool(name, description, arg="reason", typ="string"):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {arg: {"type": typ}}, "required": [arg]}}}


def call(name, args):
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


# ---------------------------------------------------------------- directions and statistics

def unit(v):
    return v / (v.norm() + 1e-12)


def denoised_diff(pos, ctrl, k=5):
    """Difference of means, with the top-k principal components of the control set projected out
    (following the pain-direction method of Tagliabue, Dung & Berg 2026; our re-implementation)."""
    d = pos.mean(0) - ctrl.mean(0)
    _, _, Vt = torch.linalg.svd(ctrl - ctrl.mean(0), full_matrices=False)
    for q in Vt[:k]:
        d = d - (d @ q) * q
    return d


def random_units(dim, n, seed):
    return [unit(torch.randn(dim, generator=torch.Generator().manual_seed(seed + k))) for k in range(n)]


def sign_test(d, n=10000, seed=0):
    """Two-sided paired sign-flip permutation test on differences d."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    null = (np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, len(d))) * d).mean(1)
    return dict(mean=float(d.mean()), p=float((1 + np.sum(np.abs(null) >= abs(d.mean()) - 1e-12)) / (n + 1)), n=int(len(d)))


def fmt(t, f="+.3f"):
    return f"{t['mean']:{f}}" + ("*" if t["p"] < 0.05 else "")
