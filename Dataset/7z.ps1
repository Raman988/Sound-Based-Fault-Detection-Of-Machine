# Create directories
New-Item -ItemType Directory -Path ".\min6dB" -Force
New-Item -ItemType Directory -Path ".\0dB" -Force
New-Item -ItemType Directory -Path ".\6dB" -Force

# Extract files from the zip archive using 7z
& 'C:\Program Files\7-Zip\7z.exe' x '.\-6_dB_fan.zip' -o'.\min6dB' -y
