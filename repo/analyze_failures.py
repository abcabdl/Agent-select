import json
import collections

# Load the report
with open(r'C:\Users\zrz20\Desktop\vscode\Agent\Agent-router\repo\data\Humaneval\humaneval_report.json', encoding='utf-8') as f:
    data = json.load(f)

# Filter failed tests
failed = [r for r in data['results'] if not r['ok']]
print(f'Total tests: {data["total"]}')
print(f'Passed: {data["passed"]}')
print(f'Failed: {len(failed)}')
print(f'Pass rate: {data["pass_rate"]:.2%}\n')

# Categorize errors
error_types = collections.Counter()
for f in failed:
    err = f.get('error', '')
    if 'AssertionError' in err:
        error_types['Test Assertion Failed (Logic Error)'] += 1
    elif 'ImportError' in err or 'Import blocked' in err:
        error_types['Import Blocked'] += 1
    elif 'IndentationError' in err:
        error_types['Indentation Error'] += 1
    elif 'SyntaxError' in err:
        error_types['Syntax Error'] += 1
    elif 'NameError' in err:
        error_types['Name Error'] += 1
    elif 'TypeError' in err:
        error_types['Type Error'] += 1
    elif 'ValueError' in err:
        error_types['Value Error'] += 1
    elif 'AttributeError' in err:
        error_types['Attribute Error'] += 1
    elif 'RecursionError' in err:
        error_types['Recursion Error'] += 1
    elif 'ZeroDivisionError' in err:
        error_types['Zero Division Error'] += 1
    elif 'KeyError' in err:
        error_types['Key Error'] += 1
    elif 'IndexError' in err:
        error_types['Index Error'] += 1
    elif 'TimeoutExpired' in err or 'timeout' in err.lower():
        error_types['Timeout'] += 1
    elif err.strip() == '':
        error_types['Empty Error (extraction_failed)'] += 1
    else:
        error_types['Other'] += 1

print('Error Type Distribution:')
for err_type, count in error_types.most_common():
    print(f'  {err_type}: {count} ({count/len(failed)*100:.1f}%)')

# Analyze extraction failures
extraction_failed_count = sum(1 for f in failed if f.get('extraction_failed'))
print(f'\nTests with extraction_failed: {extraction_failed_count}')

# Sample failed tests by category
print('\n' + '='*80)
print('SAMPLE FAILURES BY CATEGORY')
print('='*80)

categories = {
    'Test Assertion Failed (Logic Error)': [],
    'Import Blocked': [],
    'Empty Error (extraction_failed)': [],
}

for f in failed:
    err = f.get('error', '')
    if 'AssertionError' in err:
        categories['Test Assertion Failed (Logic Error)'].append(f)
    elif 'ImportError' in err or 'Import blocked' in err:
        categories['Import Blocked'].append(f)
    elif err.strip() == '':
        categories['Empty Error (extraction_failed)'].append(f)

for category, tests in categories.items():
    if tests:
        print(f'\n--- {category} (showing first 3) ---')
        for i, test in enumerate(tests[:3], 1):
            print(f'\n{i}. {test["name"]}')
            print(f'   Error: {test["error"][:300] if test["error"] else "No error message"}...')
            if test.get('extraction_failed'):
                print(f'   extraction_failed: True')
                print(f'   used_llm_fallback: {test.get("used_llm_fallback", False)}')
            # Show tool trace summary
            tool_trace = test.get('tool_trace', {})
            if tool_trace:
                for role, tools in tool_trace.items():
                    if tools:
                        print(f'   Tool attempts ({role}): {len(tools)} rounds')
                        # Check if same tool was repeated
                        tool_ids = [t.get('tool_id') for t in tools]
                        unique_tools = set(tool_ids)
                        if len(unique_tools) < len(tool_ids):
                            print(f'   WARNING: Repeated same tool(s): {list(unique_tools)}')
