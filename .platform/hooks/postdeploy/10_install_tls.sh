#!/usr/bin/env bash
set -euo pipefail

CRT_SRC="/var/app/current/deploy/tls/www.treeofknowledge.ai.crt"
KEY_SRC="/var/app/current/deploy/tls/www.treeofknowledge.ai.key"

CRT_DST="/etc/pki/tls/certs/www.treeofknowledge.ai.crt"
KEY_DST="/etc/pki/tls/private/www.treeofknowledge.ai.key"

# sanity: source files must exist in the app bundle
[ -s "$CRT_SRC" ] || { echo "Missing $CRT_SRC"; exit 1; }
[ -s "$KEY_SRC" ] || { echo "Missing $KEY_SRC"; exit 1; }

# install
sudo cp "$CRT_SRC" "$CRT_DST"
sudo cp "$KEY_SRC" "$KEY_DST"
sudo chmod 400 "$CRT_DST" "$KEY_DST"

# normalize line endings just in case
if command -v dos2unix >/dev/null 2>&1; then
  sudo dos2unix "$CRT_DST" || true
  sudo dos2unix "$KEY_DST" || true
fi

# quick validations (fail fast if bad)
sudo openssl x509 -in "$CRT_DST" -noout -subject >/dev/null
sudo openssl pkey -in "$KEY_DST" -noout >/dev/null

# cert↔key match check
CRT_HASH=$(sudo bash -c "openssl x509 -in $CRT_DST -noout -pubkey | openssl pkey -pubin -outform der | openssl sha256 | awk '{print \$2}'")
KEY_HASH=$(sudo bash -c "openssl pkey -in $KEY_DST -pubout -outform der | openssl sha256 | awk '{print \$2}'")
[ "$CRT_HASH" = "$KEY_HASH" ] || { echo "Cert/Key mismatch"; exit 1; }
