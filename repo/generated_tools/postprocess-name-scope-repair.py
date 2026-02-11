import re


def _extract_missing_symbol(error_message):
    msg = str(error_message or "")
    if not msg:
        return ""
    match = re.search(r"NameError:\s+name '([A-Za-z_][A-Za-z0-9_]*)' is not defined", msg)
    if match:
        return match.group(1)
    match = re.search(
        r"UnboundLocalError:\s+cannot access local variable '([A-Za-z_][A-Za-z0-9_]*)'",
        msg,
    )
    if match:
        return match.group(1)
    return ""


def _detect_indent(text, entry_point):
    if not entry_point:
        return ""
    for line in str(text or "").splitlines():
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return "    "


def _ensure_prefixed_block(block, indent):
    lines = []
    for line in str(block).splitlines():
        if not line.strip():
            lines.append("")
        elif line.startswith(indent):
            lines.append(line)
        else:
            lines.append(indent + line)
    return "\n".join(lines)


def run(inputs):
    payload = inputs or {}
    text = str(payload.get("completion") or "")
    entry_point = payload.get("entry_point")
    missing = _extract_missing_symbol(payload.get("error_message"))
    if not text or not missing:
        return {"completion": text, "changed": False}

    updated = text
    indent = _detect_indent(updated, entry_point)

    module_symbols = {"math", "heapq", "itertools", "re", "collections", "functools"}
    if missing in module_symbols and f"import {missing}" not in updated:
        updated = _ensure_prefixed_block(f"import {missing}", indent) + "\n" + updated

    symbol_imports = {
        "heappush": "from heapq import heappush, heappop",
        "heappop": "from heapq import heappush, heappop",
        "deque": "from collections import deque",
        "defaultdict": "from collections import defaultdict",
        "Counter": "from collections import Counter",
        "ceil": "from math import ceil",
        "sqrt": "from math import sqrt",
    }
    import_stmt = symbol_imports.get(missing)
    if import_stmt and import_stmt not in updated:
        updated = _ensure_prefixed_block(import_stmt, indent) + "\n" + updated

    helper_templates = {
        "is_prime": (
            "def is_prime(n):\n"
            "    if n <= 1:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True"
        ),
        "convert_to_float": (
            "def convert_to_float(value):\n"
            "    if isinstance(value, str):\n"
            "        value = value.replace(',', '.')\n"
            "    try:\n"
            "        return float(value)\n"
            "    except (ValueError, TypeError):\n"
            "        return None"
        ),
    }
    helper = helper_templates.get(missing)
    if helper and f"def {missing}" not in updated and f"{missing}(" in updated:
        updated = _ensure_prefixed_block(helper, indent) + "\n" + updated

    return {"completion": updated, "changed": updated != text}
