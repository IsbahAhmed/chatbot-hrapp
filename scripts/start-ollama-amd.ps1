# Start Ollama with AMD GPU on Windows (Vulkan backend).
# ChatOllama in the app talks to this server — GPU use is configured here, not in Python.
#
# 1. Install latest AMD Adrenalin drivers
# 2. Quit Ollama from the system tray first
# 3. Run: .\scripts\start-ollama-amd.ps1
# 4. In another terminal: ollama pull deepseek-coder:6.7b
# 5. Set LLM_PROVIDER=ollama in .env and restart uvicorn

$env:OLLAMA_VULKAN = "1"
# Community override if Vulkan is not picked automatically (typo is intentional in some builds):
$env:OLLAMA_GPU_OVERIDE = "vulkan"

Write-Host "OLLAMA_VULKAN=$env:OLLAMA_VULKAN"
Write-Host "Starting Ollama (AMD GPU via Vulkan)..."
Write-Host "After it starts, verify GPU in Task Manager and run: ollama run $env:OLLAMA_MODEL"
Write-Host ""

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    ollama serve
} else {
    Write-Host "ollama not in PATH. Start the Ollama app from the Start menu after setting user environment variables:"
    Write-Host "  OLLAMA_VULKAN=1"
    Write-Host "  OLLAMA_GPU_OVERIDE=vulkan"
}
