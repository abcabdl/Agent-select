from __future__ import annotations

import builtins
import multiprocessing as mp
import os
import queue
from typing import Dict


_ALLOWED_MODULES = {
    "ast",
    "astor",
    "base64",
    "bisect",  # 添加：二分查找
    "bs4",
    "collections",
    "copy",  # 添加：深拷贝/浅拷贝
    "csv",
    "cycler",
    "datetime",
    "dateutil",
    "decimal",  # 添加：精确十进制运算
    "fractions",  # 添加：分数运算
    "functools",  # 添加：函数工具
    "hashlib",
    "heapq",  # 添加：堆队列
    "httpx",
    "itertools",  # 添加：迭代工具
    "joblib",
    "json",
    "kiwisolver",
    "logging",
    "math",
    "matplotlib",
    "networkx",
    "numpy",
    "operator",  # 添加：操作符函数
    "os",
    "pandas",
    "pathlib",
    "PIL",
    "pyparsing",
    "pytz",
    "queue",  # 添加：队列
    "random",  # 添加：随机数
    "re",
    "requests",
    "scipy",
    "seaborn",
    "statsmodels",
    "sklearn",
    "statistics",
    "string",  # 添加：字符串操作
    "sympy",
    "sys",  # 添加：系统模块（安全的只读操作）
    "textwrap",  # 添加：文本包装
    "time",
    "toml",
    "tqdm",
    "typing",
    "urllib",
    "uuid",
    "xml",  # 添加：XML处理（包括xml.etree.ElementTree）
    "yaml",
    "openpyxl",
    "lxml",
    "pygame",
    "causalimpact",
    "cv2",
}
_extra_modules = {
    item.strip()
    for item in os.getenv("TOOL_ALLOWED_MODULES", "").split(",")
    if item.strip()
}
_ALLOWED_MODULES.update(_extra_modules)
_ORIGINAL_IMPORT = __import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if os.getenv("TOOL_ALLOW_ALL", "").lower() in ("1", "true", "yes"):
        return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    root = name.split(".")[0]
    extra_modules = {
        item.strip()
        for item in os.getenv("TOOL_ALLOWED_MODULES", "").split(",")
        if item.strip()
    }
    if extra_modules:
        _ALLOWED_MODULES.update(extra_modules)
    if root not in _ALLOWED_MODULES:
        raise ImportError(f"Import blocked: {name}")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


_SAFE_BUILTINS = {
    "__import__": _safe_import,
    "__build_class__": builtins.__build_class__,
    "__name__": "__tool__",
    "__file__": "<sandbox>",  # 添加：提供虚拟的 __file__ 路径
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,  # 添加：字符转换
    "compile": compile,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "ord": ord,  # 添加：字符转ASCII
    "print": print,
    "range": range,
    "reversed": reversed,  # 添加：反转迭代器
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,  # 添加：类型检查
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,  # 添加：索引错误
    "AttributeError": AttributeError,  # 添加：属性错误
}


def _execute_tool_code(code: str, inputs: dict) -> Dict:
    sandbox_globals = {"__builtins__": _SAFE_BUILTINS}
    exec(code, sandbox_globals)
    run_fn = sandbox_globals.get("run")
    if run_fn is None or not callable(run_fn):
        raise ValueError("Tool code must define callable run(inputs)")
    output = run_fn(inputs)
    if not isinstance(output, dict):
        raise TypeError("Tool run(inputs) must return dict")
    return output


def _sandbox_worker(code: str, inputs: dict, result_queue: mp.Queue) -> None:
    try:
        output = _execute_tool_code(code, inputs)
        result_queue.put({"ok": True, "output": output, "error": None})
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "output": None,
                "error": {"code": type(exc).__name__, "message": str(exc)},
            }
        )


def execute_tool_code(code: str, inputs: dict, timeout_s: float = 1.0) -> Dict:
    result_queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_sandbox_worker, args=(code, inputs, result_queue))
    process.start()
    process.join(timeout_s)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "ok": False,
            "output": None,
            "error": {"code": "timeout", "message": f"Execution exceeded {timeout_s}s"},
        }

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {
            "ok": False,
            "output": None,
            "error": {"code": "no_result", "message": "No result from sandbox"},
        }
