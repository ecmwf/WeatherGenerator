#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/nginx/weathergen-dashboard.conf.template"

SERVER_NAME="${SERVER_NAME:-_}"
STREAMLIT_UPSTREAM="${STREAMLIT_UPSTREAM:-127.0.0.1:8501}"
CERT_FILE="${CERT_FILE:-$SCRIPT_DIR/certs/dashboard.crt}"
KEY_FILE="${KEY_FILE:-$SCRIPT_DIR/certs/dashboard.key}"

if [[ ! -f "$CERT_FILE" || ! -f "$KEY_FILE" ]]; then
  cat >&2 <<EOF
Missing certificate or private key.
Generate them first, for example:

  $SCRIPT_DIR/generate_self_signed_cert.sh <public-dns-name-or-ip>

Expected:
  certificate: $CERT_FILE
  private key: $KEY_FILE
EOF
  exit 1
fi

tmp_conf="$(mktemp)"
trap 'rm -f "$tmp_conf"' EXIT

export SERVER_NAME STREAMLIT_UPSTREAM CERT_FILE KEY_FILE
uv run python - "$TEMPLATE" "$tmp_conf" <<'PY'
import os
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
replacements = {
    "__SERVER_NAME__": os.environ["SERVER_NAME"],
    "__STREAMLIT_UPSTREAM__": os.environ["STREAMLIT_UPSTREAM"],
    "__CERT_FILE__": os.environ["CERT_FILE"],
    "__KEY_FILE__": os.environ["KEY_FILE"],
}
text = template_path.read_text()
for old, new in replacements.items():
    text = text.replace(old, new)
output_path.write_text(text)
PY

if [[ -d /etc/nginx/sites-available && -d /etc/nginx/sites-enabled ]]; then
  target="/etc/nginx/sites-available/weathergen-dashboard"
  enabled="/etc/nginx/sites-enabled/weathergen-dashboard"
  sudo install -m 0644 "$tmp_conf" "$target"
  sudo ln -sfn "$target" "$enabled"
else
  target="/etc/nginx/conf.d/weathergen-dashboard.conf"
  sudo install -m 0644 "$tmp_conf" "$target"
fi

sudo nginx -t
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl reload nginx
else
  sudo service nginx reload
fi

cat <<EOF
Installed nginx configuration to:
  $target

nginx now serves HTTPS on port 443 and redirects HTTP port 80 to HTTPS.
Streamlit should be running on:
  http://$STREAMLIT_UPSTREAM
EOF
