#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_NAME="${1:-localhost}"
CERT_DIR="${2:-$SCRIPT_DIR/certs}"
DAYS="${DAYS:-365}"
CERT_FILE="$CERT_DIR/dashboard.crt"
KEY_FILE="$CERT_DIR/dashboard.key"

mkdir -p "$CERT_DIR"

if [[ "$HOST_NAME" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  SAN="IP:$HOST_NAME,IP:127.0.0.1,DNS:localhost"
else
  SAN="DNS:$HOST_NAME,DNS:localhost,IP:127.0.0.1"
fi

openssl req -x509 -newkey rsa:4096 -sha256 -nodes \
  -keyout "$KEY_FILE" \
  -out "$CERT_FILE" \
  -days "$DAYS" \
  -subj "/CN=$HOST_NAME" \
  -addext "subjectAltName=$SAN"

chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

cat <<EOF
Generated self-signed certificate:
  certificate: $CERT_FILE
  private key: $KEY_FILE

The private key is ignored by git and must not be committed.
EOF
