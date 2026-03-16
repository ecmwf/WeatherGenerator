#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [  "requests", "pyunicore" ]
# [tool.uv.sources]
# ///
"""
Create a long-lived UNICORE API token on the JSC login service.

This script authenticates to the JSC UNICORE gateway using your
JUDOOR (webservice) credentials and requests a long-lived, renewable
JWT token from the UNICORE token endpoint.

The token can then be used for automated, non-interactive access to
JSC HPC systems (JUWELS, JURECA, JUPITER, etc.) via UNICORE.

Usage:
    python create_jsc_token.py [OPTIONS]

Requirements:
    pip install requests

References:
    - https://apps.fz-juelich.de/jsc/hps/jupiter/jsctools.html#unicore-access
    - https://www.fz-juelich.de/en/jsc/services/user-support/software-tools/unicore
    - https://pyunicore.readthedocs.io/en/latest/authentication.html
    - https://unicore-docs.readthedocs.io/en/latest/user-docs/rest-api/index.html
"""

import argparse
import getpass
import json
import logging
import os
import re
import time
import sys
from datetime import datetime, timedelta, timezone
from typing import TypedDict

import requests

import pyunicore.client as uc_client
import pyunicore.credentials as uc_credentials
 
log = logging.getLogger(__name__)

# ─── JSC UNICORE endpoints ───────────────────────────────────────────
REGISTRY_URL = (
    "https://unicore.fz-juelich.de/FZJ/rest/registries/default_registry"
)

# Default token lifetime: 30 days (in seconds)
DEFAULT_LIFETIME = 30 * 24 * 3600  # 2 592 000 seconds

# Where to save the token
DEFAULT_TOKEN_FILE = os.path.expanduser("~/.jsc_unicore_token")

# {site_name: base_url}
type SiteMap = dict[str, str]


class TokenResponse(TypedDict):
    token: str


def discover_sites(username: str, password: str) -> SiteMap:
    """Query the JSC UNICORE registry and return {site_name: base_url}."""
    log.info("Querying registry: %s", REGISTRY_URL)
    resp = requests.get(
        REGISTRY_URL,
        auth=(username, password),
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()

    sites: SiteMap = {}
    for entry in data.get("entries", []):
        href = entry.get("href", "")
        if entry.get("type", "") == "TargetSystemFactory":
            # Extract the base /rest/core URL and site name
            m = re.match(r"(https://\S+/rest/core).*", href)
            n = re.match(r"https://\S+/(\S+)/rest/core", href)
            if m and n:
                sites[n.group(1)] = m.group(1)
    return sites


def create_token(
    base_url: str,
    username: str,
    password: str,
    lifetime: int,
    limited: bool,
    renewable: bool,
) -> TokenResponse:
    """
    Create a JWT token via pyunicore's Client.issue_auth_token().

    Uses UsernamePassword credential → Client → issue_auth_token(), which
    is pyunicore's own token-issuing path and handles all UNICORE-specific
    headers (security sessions, preferences, Accept: text/plain).
    """
    log.info(
        "Requesting token from: %s  (lifetime=%ds ~%dd, limited=%s, renewable=%s)",
        base_url, lifetime, lifetime // 86400, limited, renewable,
    )

    credential = uc_credentials.UsernamePassword(username, password)
    client = uc_client.Client(credential, site_url=base_url)
    log.debug("Client created for: %s", base_url)

    token = client.issue_auth_token(
        lifetime=lifetime,
        renewable=renewable,
        limited=limited,
    )
    log.debug("Token (first 80 chars): %r", token[:80])
    return TokenResponse(token=token)


def discover_sites_bearer(token: str) -> SiteMap:
    """Query the registry using a Bearer token; return {site_name: base_url}."""
    log.info("Querying registry: %s", REGISTRY_URL)
    resp = requests.get(
        REGISTRY_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    resp.raise_for_status()
    data = resp.json()

    sites: SiteMap = {}
    for entry in data.get("entries", []):
        href = entry.get("href", "")
        if entry.get("type", "") == "TargetSystemFactory":
            m = re.match(r"(https://\S+/rest/core).*", href)
            n = re.match(r"https://\S+/(\S+)/rest/core", href)
            if m and n:
                sites[n.group(1)] = m.group(1)
    return sites

def run_command(client: uc_client.Client, command: str, project: str | None,
                poll_interval: float) -> None:
    """Submit a non-batch (interactive) job that runs `command` and stream output."""
    job_desc = {
        "Executable": command,
        "Environment": {"UC_PREFER_INTERACTIVE_EXECUTION": "true"},
    }
    if project:
        job_desc["Project"] = project
 
    print(f"  Submitting: {command}")
    job = client.new_job(job_description=job_desc)
    print(f"  Job URL:    {job.resource_url}")
    print(f"  Status:     {job.properties['status']}")
 
    # Poll until finished
    while job.properties["status"] not in ("SUCCESSFUL", "FAILED", "DONE"):
        time.sleep(poll_interval)
        # Force refresh
        job.properties  # noqa – property access triggers refresh
 
    status = job.properties["status"]
    print(f"  Final:      {status}")
 
    # Read stdout / stderr from the working directory
    wd = job.working_dir
    print("\n--- stdout ---")
    try:
        stdout = wd.stat("/stdout")
        print(stdout.raw().read().decode("utf-8", errors="replace"))
    except Exception:
        print("  (empty or unavailable)")
 
    stderr_text = ""
    try:
        stderr_file = wd.stat("/stderr")
        stderr_text = stderr_file.raw().read().decode("utf-8", errors="replace")
    except Exception:
        pass
 
    if stderr_text.strip():
        print("--- stderr ---")
        print(stderr_text)
 
    # Clean up the job on the server
    try:
        job.delete()
    except Exception:
        pass
 
    if status == "FAILED":
        sys.exit(1)
 
 

def verify_token(base_url: str, token: str) -> bool:
    """
    Verify that *token* is accepted by *base_url*.

    Logs identity info and returns True on success, False otherwise.
    """
    log.info("Checking token against: %s", base_url)
    try:
        resp = requests.get(
            base_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )
    except requests.exceptions.ConnectionError as e:
        log.error("Connection failed: %s", e)
        return False

    log.debug("HTTP status: %s", resp.status_code)
    if not resp.ok:
        log.error("Server rejected the token (HTTP %s).", resp.status_code)
        log.error("Server response: %s", resp.text[:300])
        return False

    info = resp.json()
    client_info = info.get("client", {})
    dn: str = client_info.get("dn", "N/A")
    role: str = client_info.get("role", {}).get("selected", "N/A")
    log.info("Status: OK")
    log.info("DN:     %s", dn)
    log.info("Role:   %s", role)
    return True


def save_token(token: str, path: str) -> None:
    """Save the token string to a file with restricted permissions."""
    with open(path, "w") as f:
        f.write(token)
    os.chmod(path, 0o600)
    log.info("Token saved to: %s  (permissions: 600)", path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a long-lived UNICORE API token for JSC systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Interactive — prompts for username and password:
  python create_jsc_token.py

  # Specify username (password still prompted securely):
  python create_jsc_token.py -u myuser

  # Custom lifetime of 90 days, save to custom path:
  python create_jsc_token.py -u myuser --lifetime 7776000 -o /tmp/my_token

  # Choose a specific site (skip interactive selection):
  python create_jsc_token.py -u myuser --site JURECA

  # List available sites only:
  python create_jsc_token.py -u myuser --list-sites
""",
    )
    parser.add_argument(
        "-u", "--username",
        help="JUDOOR username (if omitted, will be prompted)",
    )
    parser.add_argument(
        "--lifetime",
        type=int,
        default=DEFAULT_LIFETIME,
        help=f"Token lifetime in seconds (default: {DEFAULT_LIFETIME} = 30 days)",
    )
    parser.add_argument(
        "--limited",
        action="store_true",
        default=False,
        help="Make the token valid only for the chosen UNICORE/X server",
    )
    parser.add_argument(
        "--no-renewable",
        action="store_true",
        default=False,
        help="Disable token renewal (by default tokens are renewable)",
    )
    parser.add_argument(
        "--site",
        help="Target site name (e.g. JURECA, JUWELS, JUPITER). "
             "If omitted, you can choose interactively.",
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="List available sites and exit",
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_TOKEN_FILE,
        help=f"File to save the token to (default: {DEFAULT_TOKEN_FILE})",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the token to stdout instead of saving to a file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that the saved token (see -o/--output) is still valid. "
             "No credentials required. Use --site to target a specific site.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging (shows raw HTTP response details)",
    )
    parser.add_argument(
        "-p", "--project",
        help="Budget/project account to charge (passed as UNICORE 'Project')",
    )
    parser.add_argument(
        "command", nargs="?",
        help="Shell command to execute (e.g. 'squeue -u $USER')",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(filename)s:%(lineno)d: %(message)s",
    )

    # ── --check: verify existing token, no credentials needed ─────────
    if args.check:
        token_path = args.output
        if not os.path.exists(token_path):
            log.error("Token file not found: %s", token_path)
            sys.exit(1)
        with open(token_path) as f:
            token = f.read().strip()
        if not token:
            log.error("Token file is empty: %s", token_path)
            sys.exit(1)
        log.info("Token file: %s", token_path)

        try:
            sites = discover_sites_bearer(token)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            log.error(
                "Registry query failed (HTTP %s). Token may be expired or invalid.",
                status,
            )
            sys.exit(1)
        except requests.exceptions.ConnectionError as e:
            log.error("Could not connect to registry: %s", e)
            sys.exit(1)

        if not sites:
            log.error("No UNICORE sites found in registry.")
            sys.exit(1)

        if args.site:
            if args.site not in sites:
                log.error(
                    "Site '%s' not found. Available: %s",
                    args.site, ", ".join(sorted(sites)),
                )
                sys.exit(1)
            to_check: SiteMap = {args.site: sites[args.site]}
        else:
            to_check = sites

        log.info("Found %d site(s); checking %d:", len(sites), len(to_check))
        ok = True
        for name, url in sorted(to_check.items()):
            log.info("[%s]", name)
            # BearerToken sends "Authorization: Bearer <token>" — correct for
            # a pre-issued UNICORE JWT.  JWTToken is for *generating* tokens
            # from a private key; OIDCToken is for OAuth2/OIDC flows.
            credential = uc_credentials.BearerToken(token=token)
            client = uc_client.Client(credential, url)
            props = client.properties
            client_info = props.get("client", {})
            dn: str = client_info.get("dn", "N/A")
            role: str = client_info.get("role", {}).get("selected", "N/A")
            log.info("  DN:   %s", dn)
            log.info("  Role: %s", role)
            if role == "anonymous":
                log.error("  Token has no privileges on %s (role is anonymous).", name)
                ok = False
        sys.exit(0 if ok else 1)

    # ── Gather credentials ────────────────────────────────────────────
    log.info("JSC UNICORE Long-Lived Token Creator")
    log.info("Authenticate with your JUDOOR (webservice) credentials.")

    username = args.username or input("Username: ")
    password = getpass.getpass("Password: ")

    # ── Discover sites from registry ──────────────────────────────────
    try:
        sites = discover_sites(username, password)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            log.error("Authentication failed. Check your JUDOOR username/password.")
        else:
            log.error("%s", e)
        sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        log.error("Could not connect to registry: %s", e)
        sys.exit(1)

    if not sites:
        log.error("No UNICORE sites found in registry.")
        sys.exit(1)

    log.info("Found %d site(s):", len(sites))
    for i, name in enumerate(sorted(sites), 1):
        log.info("  [%d] %s  —  %s", i, name, sites[name])

    if args.list_sites:
        sys.exit(0)

    # ── Select a site ─────────────────────────────────────────────────
    sorted_names = sorted(sites)
    if args.site:
        if args.site not in sites:
            log.error(
                "Site '%s' not found. Available: %s",
                args.site, ", ".join(sorted_names),
            )
            sys.exit(1)
        chosen = args.site
    else:
        choice = input(f"Select site [1-{len(sorted_names)}] (default: 1): ").strip()
        idx = int(choice) - 1 if choice else 0
        if idx < 0 or idx >= len(sorted_names):
            log.error("Invalid selection.")
            sys.exit(1)
        chosen = sorted_names[idx]

    base_url = sites[chosen]
    log.info("Selected site: %s", chosen)

    # ── Create the token ──────────────────────────────────────────────
    renewable = not args.no_renewable
    try:
        result = create_token(
            base_url=base_url,
            username=username,
            password=password,
            lifetime=args.lifetime,
            limited=args.limited,
            renewable=renewable,
        )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        log.error("Token creation failed (HTTP %s).", status)
        if e.response is not None:
            try:
                log.error("Server response: %s", e.response.text[:500])
            except Exception:
                pass
        sys.exit(1)

    token = result["token"]

    # ── Output ────────────────────────────────────────────────────────
    expiry = datetime.now(timezone.utc) + timedelta(seconds=args.lifetime)
    log.info("Token created successfully!")
    log.info("  Site:      %s", chosen)
    log.info("  Renewable: %s", renewable)
    log.info("  Limited:   %s", args.limited)
    log.info("  Lifetime:  %ds (~%d days)", args.lifetime, args.lifetime // 86400)
    log.info("  Expires:   ~%s", expiry.strftime("%Y-%m-%d %H:%M UTC"))

    if args.print_only:
        print(token)
    else:
        save_token(token, args.output)
        log.info("To use with pyunicore:")
        log.info("  import pyunicore.credentials as uc_credentials")
        log.info("  credential = uc_credentials.OIDCToken(token=open('%s').read().strip())", args.output)
        log.info("To use with UCC (preferences file):")
        log.info("  authentication-method=bearer-token")
        log.info("  token=<contents of %s>", args.output)
        log.info("To use with curl:")
        log.info('  curl -H "Authorization: Bearer $(cat %s)" \\', args.output)
        log.info('       -H "Accept: application/json" \\')
        log.info('       %s', base_url)

    # ── Quick verification ────────────────────────────────────────────
    print(base_url, token)
    verify_token(base_url, token)


if __name__ == "__main__":
    main()
