if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Running uv sync..."
    uv sync
    uv pip install git+https://github.com/lucasb-eyer/pydensecrf.git
}
else {
    Write-Host "Please install uv to setup the environment"
    Write-Host "Hint: Run 'pip install uv' in PowerShell."
}