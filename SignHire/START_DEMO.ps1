# SignHire — One-click demo launcher
# Double-click this file or run it in PowerShell before your presentation.
# Make sure your HP camera is connected and Chrome/Edge is installed.

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SignHire — Starting Demo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project folder
Set-Location "C:\Users\Aradhy Bhardwaj\OneDrive\Desktop\Forma-AI\SignHire"

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Open browser automatically after 8 seconds (time for model to load)
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 8
    Start-Process "http://localhost:5000"
} | Out-Null

# Start the Flask app
Write-Host "Loading model and starting server..." -ForegroundColor Yellow
Write-Host "Browser will open automatically at http://localhost:5000" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the demo." -ForegroundColor Gray
Write-Host ""

python src/app/app.py --camera 1 --model baseline
