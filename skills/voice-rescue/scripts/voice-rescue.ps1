# voice-rescue launcher for Windows PowerShell (untested in v0.1.0; see README).
$ErrorActionPreference = "Stop"
$skill = Split-Path -Parent $PSScriptRoot
$data = if ($env:VOICE_RESCUE_HOME) { $env:VOICE_RESCUE_HOME } else { Join-Path $env:LOCALAPPDATA "voice-rescue" }
New-Item -ItemType Directory -Force -Path $data | Out-Null
$env:UV_PROJECT_ENVIRONMENT = Join-Path $data "env"
$env:PYTHONUNBUFFERED = "1"
uv run --quiet --frozen --no-dev --project $skill --python 3.12 --python-preference only-managed voice-rescue @args
exit $LASTEXITCODE
