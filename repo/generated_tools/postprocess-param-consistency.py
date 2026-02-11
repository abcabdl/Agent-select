import re


def _normalize_param_names(param_names):
    if not isinstance(param_names, list):
        return []
    normalized = []
    for item in param_names:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _auto_fix_parameter_consistency(completion, param_names):
    text = str(completion or "")
    if not text or not param_names:
        return text

    identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text))
    if not identifiers:
        return text

    if len(param_names) == 1:
        target = param_names[0]
        if target not in identifiers:
            aliases = ["lst", "arr", "array", "nums", "values", "s", "string", "text", "sentence", "grid", "x"]
            for alias in aliases:
                if alias in identifiers and alias != target:
                    text = re.sub(rf"\b{re.escape(alias)}\b", target, text)
                    break

    if len(param_names) >= 2:
        p1 = param_names[0]
        p2 = param_names[1]
        if p1 not in identifiers and p2 not in identifiers and "a" in identifiers and "b" in identifiers:
            if p1 != "a":
                text = re.sub(r"\ba\b", p1, text)
            if p2 != "b":
                text = re.sub(r"\bb\b", p2, text)

    return text


def run(inputs):
    payload = inputs or {}
    text = str(payload.get("completion") or "")
    params = _normalize_param_names(payload.get("param_names"))
    if not params:
        return {"completion": text, "changed": False}
    updated = _auto_fix_parameter_consistency(text, params)
    return {"completion": updated, "changed": updated != text}

