#!/usr/bin/env bash
set -e
sudo nginx -t
sudo systemctl reload nginx
