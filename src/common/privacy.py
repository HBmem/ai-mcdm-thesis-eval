"""Sanitize exported/displayed data before it crosses the audit/UI boundary."""
import os
import re

SENSITIVE = re.compile(r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|email|participant[_-]?name)", re.I)


def sanitize(value, secrets=()):
    known = tuple(s for s in (*secrets, os.environ.get("OLLAMA_API_KEY", ""), os.environ.get("OPENAI_API_KEY", "")) if s)
    def clean(item):
        if isinstance(item, dict):
            return {k: "[REDACTED]" if SENSITIVE.search(str(k)) else clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, str):
            for secret in known:
                item = item.replace(secret, "[REDACTED]")
            item = re.sub(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", item, flags=re.I)
            item = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", item)
            item = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[REDACTED]", item)
            item = re.sub(r'(?i)(api[_-]?key|password|secret|access[_-]?token)([\s"\x27:=]+)([^\s,"\x27}]+)', r'\1\2[REDACTED]', item)
        return item
    return clean(value)