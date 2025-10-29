#!/usr/bin/env bash
set -euo pipefail

sudo nginx -t

# ensure it starts if inactive; reload if already running
if systemctl is-active --quiet nginx; then
  sudo systemctl reload nginx
else
  sudo systemctl enable nginx
  sudo systemctl start nginx
fi
