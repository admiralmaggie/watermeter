#!/bin/bash

# Simple systemd service setup 
set -e

APP_DIR="$(pwd)"
SERVICE_NAME="meter-monitor-service"
USER="$(whoami)"

echo "Setting up systemd service..."

# Create systemd service file
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Meter Monitor Service
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/venv/bin/python ${APP_DIR}/monitor_meter.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Set secure permissions
chmod 600 creds.txt 2>/dev/null || true

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start ${SERVICE_NAME}

echo "Service ${SERVICE_NAME} created and started"
echo "Status: sudo systemctl status ${SERVICE_NAME}"
echo "Logs: sudo journalctl -u ${SERVICE_NAME} -f"
