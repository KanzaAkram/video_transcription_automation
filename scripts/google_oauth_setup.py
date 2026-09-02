#!/usr/bin/env python
"""Get a Google Drive refresh token for `oauth_user` mode.

    python scripts/google_oauth_setup.py --client-secrets credentials/oauth_client.json

Opens a browser, asks you to grant access to your own Drive, and prints the
three lines to paste into `.env`. Run once; the refresh token does not expire
so long as the OAuth app is published (see below).

Why this exists rather than a service account
---------------------------------------------
A service account is the simpler credential and it is the wrong one for a
personal Gmail account, for a reason that only shows up after everything looks
set up: **a service account has no Drive storage of its own.** It can rename
and move files someone shared with it, because neither creates storage. It
cannot upload — the new file would be owned by the service account, and the
API answers `storageQuotaExceeded`.

This pipeline uploads a transcript beside every recording, so on a personal
account the service-account path fails at the last stage of the first lecture
it processes. Google Workspace does not have this problem, because files in a
Shared Drive are owned by the drive rather than by the uploader.

So: Workspace with a Shared Drive -> service account. Personal Gmail -> this.

Two things to do in the Cloud Console first
-------------------------------------------
1. **Enable the Drive API.** APIs & Services -> Library -> "Google Drive API"
   -> Enable. Creating credentials does not do this, and the failure it
   produces later is a 403 that does not mention the API being off.

2. **Publish the OAuth app.** APIs & Services -> OAuth consent screen ->
   Publish app. While the app is in *Testing*, Google expires refresh tokens
   after **seven days** — an automation set up on Monday stops the following
   Monday, silently, and the error looks like a credentials problem rather than
   an expiry. Publishing removes the limit. You will see an "unverified app"
   warning on the consent screen; that is expected, and verification is only
   required past 100 users.

Then: Credentials -> Create credentials -> OAuth client ID -> **Desktop app**,
download the JSON, and point this script at it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:  # pragma: no cover - import guard
    print(
        'The Google OAuth libraries are not installed.  pip install -e ".[cloud]"',
        file=sys.stderr,
    )
    raise SystemExit(1) from None

from pipeline.storage.google_drive import SCOPES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--client-secrets",
        type=Path,
        default=Path("credentials/oauth_client.json"),
        help="The OAuth client JSON downloaded from the Cloud Console "
        "(default: credentials/oauth_client.json)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local port for the consent redirect. 0 picks a free one, which is "
        "what a Desktop-app client expects (default: 0)",
    )
    args = parser.parse_args()

    if not args.client_secrets.exists():
        print(
            f"No such file: {args.client_secrets}\n\n"
            "Download it from the Cloud Console: APIs & Services -> Credentials ->\n"
            "Create credentials -> OAuth client ID -> Desktop app -> download JSON.",
            file=sys.stderr,
        )
        return 1

    # Read the client id and secret out of the file rather than asking for them
    # separately: they are already in it, and re-typing a secret is a good way
    # to spend an afternoon on a typo.
    payload = json.loads(args.client_secrets.read_text(encoding="utf-8"))
    section = payload.get("installed") or payload.get("web") or {}
    client_id = section.get("client_id", "")
    client_secret = section.get("client_secret", "")
    if not client_id:
        print(
            f"{args.client_secrets} does not look like an OAuth client file — it has "
            'no "installed" or "web" section. Make sure you downloaded the OAuth '
            "client ID, not the service-account key.",
            file=sys.stderr,
        )
        return 1

    print("Scopes requested:", " ".join(SCOPES))
    print("A browser window will open. Sign in as the account that OWNS the")
    print("recordings, and accept the 'unverified app' warning.\n")

    flow = InstalledAppFlow.from_client_secrets_file(str(args.client_secrets), SCOPES)
    # access_type=offline is what makes Google return a refresh token at all,
    # and prompt=consent forces one even if this account has authorised the app
    # before — without it a second run yields a token with no refresh token and
    # a confusing "None" below.
    credentials = flow.run_local_server(
        port=args.port, access_type="offline", prompt="consent"
    )

    if not credentials.refresh_token:
        print(
            "\nGoogle returned no refresh token. That happens when the account has "
            "already granted this app access. Revoke it at "
            "https://myaccount.google.com/permissions and run this again.",
            file=sys.stderr,
        )
        return 1

    print("\n" + "=" * 72)
    print("Paste these into .env:\n")
    print("GOOGLE_AUTH_MODE=oauth_user")
    print(f"GOOGLE_OAUTH_CLIENT_ID={client_id}")
    print(f"GOOGLE_OAUTH_CLIENT_SECRET={client_secret}")
    print(f"GOOGLE_OAUTH_REFRESH_TOKEN={credentials.refresh_token}")
    print("=" * 72)
    print(
        "\nThe refresh token is a long-lived credential for your whole Drive. "
        "Treat it like a password: .env is gitignored, keep it that way.\n"
        "If the OAuth app is still in 'Testing', publish it — otherwise this "
        "token stops working in seven days."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
