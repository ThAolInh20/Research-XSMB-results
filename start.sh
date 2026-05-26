#!/bin/bash

# Dừng và xóa containers
docker compose down

# Khởi động containers ở chế độ detached
docker compose up -d

# Khởi động cloudflared tunnel
cloudflared tunnel --url http://localhost:5678

# Chạy lệnh dự đoán
python execute.py
