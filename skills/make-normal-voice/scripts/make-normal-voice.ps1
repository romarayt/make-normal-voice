# make-normal-voice launcher for Windows PowerShell (untested in v0.2.0; see README).
$ErrorActionPreference = "Stop"
$skill = Split-Path -Parent $PSScriptRoot
$data = if ($env:MAKE_NORMAL_VOICE_HOME) { $env:MAKE_NORMAL_VOICE_HOME } else { Join-Path $env:LOCALAPPDATA "make-normal-voice" }
New-Item -ItemType Directory -Force -Path $data | Out-Null
$env:UV_PROJECT_ENVIRONMENT = Join-Path $data "env"
$env:PYTHONUNBUFFERED = "1"
uv run --quiet --frozen --no-dev --project $skill --python 3.12 --python-preference only-managed make-normal-voice @args
exit $LASTEXITCODE
