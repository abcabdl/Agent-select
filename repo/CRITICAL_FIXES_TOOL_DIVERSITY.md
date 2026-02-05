# Critical Fixes: Tool Diversity Mechanism

## Date: 2026-02-03

## Problem Analysis from Test Results (20 tests, 80% pass rate)

### Root Cause: Tool Diversity Mechanism Was BROKEN

The forced tool diversity mechanism at line 768 was **NOT working** because:

**All tools share the same base category:**
- Tool IDs: `code-generation-assemblesnippets`, `code-generation-generatemath6`, `code-generation-generateedgecase12`
- Old extraction logic: `tool_id.rsplit('-', 1)[0]` → Always extracted `'code-generation'`
- Result: All tools counted as the same category, so diversity never triggered!

### Evidence from Debug Output

**HumanEval_151_double_the_difference:**
```
[DEBUG] Tool category 'code-generation' failures: 10, consecutive: 10
```
- Used only `assemblesnippets` tools 10 rounds straight
- All rounds generated identical wrong code: `isinstance(x, int)` filters out 5.0
- **Diversity mechanism never triggered because all tools were counted as 'code-generation' category**

**HumanEval_160_do_algebra:**
- All 5 tools tried were `generatemath` variants (generatemath6, generatemath18, generatemath30, generatemath42, generatemath54)
- All failed with same ImportError: `Import blocked: decimal`
- **Should have switched to different tool family after 3 failures**

## Fixes Implemented

### Fix 1: Correct Tool Category Extraction (Line ~820)

**Before:**
```python
tool_category = tool_id.rsplit('-', 1)[0] if '-' in tool_id else tool_id
# Extracted: 'code-generation' for ALL tools
```

**After:**
```python
parts = tool_id.split('-')
if len(parts) >= 3:
    # Extract: 'assemblesnippets', 'generatemath', 'generateedgecase'
    tool_family = parts[2]
    # Remove numbers: 'generatemath6' -> 'generatemath'
    import re
    tool_category = re.sub(r'\d+$', '', tool_family)
else:
    tool_category = tool_id.rsplit('-', 1)[0] if '-' in tool_id else tool_id
```

**Now extracts:**
- `code-generation-assemblesnippets12` → `assemblesnippets`
- `code-generation-generatemath6` → `generatemath`
- `code-generation-generateedgecase12` → `generateedgecase`
- `code-generation-generatefunctionbody` → `generatefunctionbody`

### Fix 2: Update Forced Diversity Logic (Line ~770)

**重要说明：只在当前agent的工具范围内切换！**

`tool_ids` 来自 `agent_context.get("available_tool_ids")`，是当前agent可用的工具列表。
强制多样性只会在这个范围内选择不同类别的工具，**不会跨agent切换**。

**Before:**
```python
for tid in tool_ids:
    tid_category = tid.rsplit('-', 1)[0]  # Always 'code-generation'
    if tid_category != tool_category:     # Never true!
        different_category_tools.append(tid)
```

**After:**
```python
for tid in tool_ids:
    parts = tid.split('-')
    if len(parts) >= 3:
        tool_family = parts[2]
        import re
        tid_category = re.sub(r'\d+$', '', tool_family)
    else:
        tid_category = tid.rsplit('-', 1)[0] if '-' in tid else tid
    
    if tid_category != tool_category:  # Now properly compares tool families
        different_category_tools.append(tid)
```

### Fix 3: Track Category Changes (Line ~835)

**Added:**
```python
# Track consecutive same-category usage (update last_tool_category)
if last_tool_category and last_tool_category != tool_category:
    # Category changed, reset consecutive counter
    consecutive_same_category_failures = 0
    print(f"[DEBUG] Tool category changed from '{last_tool_category}' to '{tool_category}', resetting consecutive failures", file=sys.stderr)
last_tool_category = tool_category
```

### Fix 4: More Aggressive isinstance Error Detection (Line ~706)

**Before:**
```python
if "isinstance" in failed_code and ("5.0" in last_test_error or "float" in error_lower):
```

**After:**
```python
if "isinstance" in failed_code and "int" in failed_code:
    if "5.0" in last_test_error or "float" in error_lower or "assertionerror" in error_lower:
        diagnostic_hints.append(
            "❌ TYPE CHECK ERROR: Using isinstance(x, int) filters out floats like 5.0 that are mathematically integers.\n"
            "✅ CRITICAL FIX: Replace isinstance(x, int) with: isinstance(x, (int, float)) and x == int(x)\n"
            "   This accepts both 5 and 5.0 as valid integers."
        )
```

## Expected Improvements

### HumanEval_151_double_the_difference
**Before:** 10 rounds, all `assemblesnippets`, all same wrong code
**After:** 
- Round 0-2: `assemblesnippets` with isinstance error
- Round 3: **Switch to `generatemath`** (forced diversity triggers)
- Should fix isinstance bug with stronger diagnostic hint

### HumanEval_160_do_algebra
**Before:** 5 rounds, all `generatemath` variants, all ImportError
**After:**
- Round 0-2: `generatemath` tools hit ImportError
- Round 3: **Switch to `assemblesnippets` or `generatefunctionbody`** (forced diversity)
- Different tool category won't try to import decimal

### Overall Impact
- Tool diversity mechanism now **actually works**
- After 3 consecutive same-category failures → forced switch to different tool family
- Debug output will show: `"[DEBUG] Forcing tool category switch from 'assemblesnippets' after 3 failures"`
- Should reduce repetitive failures, improve pass rate

## Verification Steps

**重要提醒：工具切换只在当前agent的可用工具范围内进行！**

`tool_ids` 来自 `agent_context.get("available_tool_ids")`，确保：
- ✅ 只在当前agent配对的工具中切换类别
- ✅ 不会跨越到其他agent的工具
- ✅ 如果agent只有单一类别工具，debug会提示"Cannot switch category"

1. Run 20 tests again and check debug output for:
   - `"[DEBUG] Tool category changed from X to Y"`
   - `"[DEBUG] Forcing tool category switch from X after 3 failures"`
   
2. Check HumanEval_151 specifically:
   - Should see category switch after 3 failures
   - Should see stronger isinstance diagnostic hint
   
3. Check HumanEval_160:
   - Should switch away from generatemath after 3 ImportErrors
   - Should try assemblesnippets or other categories

## Known Remaining Issues

1. **HumanEval_105_by_length** - Wrong role selected (planner vs builder)
   - Uses `lst` parameter but function signature has `arr`
   - Need to fix role selection or parameter name extraction

2. **HumanEval_96_count_up_to** - Parameter name mismatch
   - Uses `num` inside code but parameter is `n`
   - Need better parameter name constraint enforcement

3. **LLM fallback failures** - Some tasks fall through to LLM fallback
   - Need to investigate why all tools fail
   - May need better tool selection prompts
