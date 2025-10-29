#!/usr/bin/env bash
set -euo pipefail

# Source files committed in your repo
CRT_SRC="/var/app/current/deploy/tls/www.treeofknowledge.ai.crt"
KEY_SRC="/var/app/current/deploy/tls/www.treeofknowledge.ai.key"

# Destination paths used by nginx.conf
CRT_DST="/etc/pki/tls/certs/www.treeofknowledge.ai.crt"
KEY_DST="/etc/pki/tls/private/www.treeofknowledge.ai.key"

# Sanity: repo files must exist
[ -s "$CRT_SRC" ] || { echo "Missing $CRT_SRC"; exit 1; }
[ -s "$KEY_SRC" ] || { echo "Missing $KEY_SRC"; exit 1; }

# Install certificate (full chain; leaf first)
sudo cp "$CRT_SRC" "$CRT_DST"

# Always convert the key to **PKCS#1 (RSA)** for nginx/OpenSSL 3 compatibility
# This covers both PKCS#1 and PKCS#8 inputs safely.
sudo openssl rsa -in "$KEY_SRC" -out "$KEY_DST"

# Tight perms
sudo chmod 400 "$CRT_DST" "$KEY_DST"

# Normalize line endings (no-op if already LF)
if command -v dos2unix >/dev/null 2>&1; then
  sudo dos2unix "$CRT_DST" || true
  sudo dos2unix "$KEY_DST" || true
fi

# Quick parse checks (fail fast if malformed)
sudo openssl x509 -in "$CRT_DST" -noout -subject >/dev/null
sudo openssl pkey -in "$KEY_DST" -noout >/dev/null

# Cert↔key match (public key hash must match)
CRT_HASH=$(sudo bash -c "openssl x509 -in $CRT_DST -noout -pubkey | openssl pkey -pubin -outform der | openssl sha256 | awk '{print \$2}'")
KEY_HASH=$(sudo bash -c "openssl pkey -in $KEY_DST -pubout -outform der | openssl sha256 | awk '{print \$2}'")
[ "$CRT_HASH" = "$KEY_HASH" ] || { echo "Cert/Key mismatch"; exit 1; }
