# 临时允许所有导入（用于测试）
# 使用方法：. .\allow_all_imports.ps1

$env:TOOL_ALLOW_ALL = "true"
Write-Host "✓ 已允许所有模块导入 (TOOL_ALLOW_ALL=true)" -ForegroundColor Green
Write-Host "⚠ 仅用于测试！生产环境请使用白名单模式" -ForegroundColor Yellow

# 或者只添加特定模块：
# $env:TOOL_ALLOWED_MODULES = "sys,subprocess,threading"
