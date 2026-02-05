"""
展示每种工具类型的独特特性
"""
from pathlib import Path
import re

def show_tool_differences():
    """展示工具之间的差异"""
    tools_dir = Path(__file__).parent / "generated_tools"
    
    # 选择每种类型的代表文件
    examples = {
        "assemblesnippets": "code-generation-assemblesnippets.py",
        "generatealgorithm": "code-generation-generatealgorithm1.py",
        "generatedatastructure": "code-generation-generatedatastructure2.py",
        "generatedp": "code-generation-generatedp5.py",
        "generateedgecase": "code-generation-generateedgecase12.py",
        "generatefunctionbody": "code-generation-generatefunctionbody.py",
        "generategraph": "code-generation-generategraph4.py",
        "generategreedy": "code-generation-generategreedy8.py",
        "generateio": "code-generation-generateio10.py",
        "generatemath": "code-generation-generatemath6.py",
        "generatemoduleskeleton": "code-generation-generatemoduleskeleton.py",
        "generateparsing": "code-generation-generateparsing7.py",
        "generaterecursion": "code-generation-generaterecursion9.py",
        "generaterobustness": "code-generation-generaterobustness11.py",
        "generatestring": "code-generation-generatestring3.py",
    }
    
    print("=" * 80)
    print("各工具类型的独特特性")
    print("=" * 80)
    
    for tool_type, filename in examples.items():
        filepath = tools_dir / filename
        if not filepath.exists():
            continue
        
        content = filepath.read_text('utf-8')
        
        # 提取system prompt
        system_match = re.search(r'system = """(.+?)"""', content, re.DOTALL)
        if system_match:
            system_prompt = system_match.group(1).strip()
            # 只显示前两行
            lines = system_prompt.split('\n')
            preview = '\n    '.join(lines[:3])
            
            print(f"\n📦 {tool_type}")
            print(f"   {preview}")
            if len(lines) > 3:
                print(f"   ... (共{len(lines)}行规则)")
        
        # 检查额外的imports
        imports = []
        if "from collections import" in content:
            imports.append("collections")
        if "import math" in content:
            imports.append("math")
        if "from functools import lru_cache" in content:
            imports.append("@lru_cache")
        if "import json" in content:
            imports.append("json")
        if "import csv" in content:
            imports.append("csv")
        
        if imports:
            print(f"   特殊依赖: {', '.join(imports)}")
    
    print(f"\n{'='*80}")
    print("总结：15种工具类型，每种都有专门的system prompt和处理逻辑")
    print("=" * 80)

if __name__ == "__main__":
    show_tool_differences()
