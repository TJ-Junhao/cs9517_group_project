if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Running uv sync..."
    uv sync
} else {
    Write-Host "Please install uv to setup the environment"
    Write-Host "Hint: Run 'pip install uv' in PowerShell."
}