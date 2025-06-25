$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$zipPath = "$env:TEMP\ffmpeg.zip"
$installDir = "$env:LOCALAPPDATA\ffmpeg"

if (Test-Path "$installDir\bin\ffmpeg.exe") {
    Write-Host "FFmpeg is already installed at $installDir"
    exit
}

Write-Host "Downloading FFmpeg..."
Invoke-WebRequest -Uri $ffmpegUrl -OutFile $zipPath

Write-Host "Extracting FFmpeg..."
Expand-Archive -Path $zipPath -DestinationPath $installDir -Force

# get the actual folder inside the zip
$subfolder = Get-ChildItem -Directory $installDir | Select-Object -First 1
$binPath = "$($subfolder.FullName)\bin"

# add to PATH for current user
$envPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($envPath -notlike "*$binPath*") {
    [Environment]::SetEnvironmentVariable("Path", "$envPath;$binPath", "User")
    Write-Host "FFmpeg path added to User PATH. You may need to restart your terminal."
} else {
    Write-Host "FFmpeg path already in PATH."
}