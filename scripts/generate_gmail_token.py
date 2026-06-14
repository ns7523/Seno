from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.integrations.google_auth import GOOGLE_SCOPES, GoogleAuth


def main() -> None:
    force = "--force" in sys.argv[1:]
    default_client_secret = Path("app/gmail/client_secret.json")
    root_client_secret = Path("client_secret.json")
    client_secret = root_client_secret if root_client_secret.exists() else default_client_secret
    token_file = Path("token.json")

    if not client_secret.exists():
        raise SystemExit(
            "Missing Google OAuth client secret. Put it at client_secret.json "
            "or app/gmail/client_secret.json, then rerun this command."
        )

    auth = GoogleAuth(str(client_secret), str(token_file), GOOGLE_SCOPES)
    creds = auth.regenerate_credentials() if force else auth.get_credentials()
    if not creds.valid:
        raise SystemExit("Google OAuth failed: generated credentials are invalid.")
    print(f"Generated shared Gmail + Calendar token: {token_file.resolve()}")


if __name__ == "__main__":
    main()
