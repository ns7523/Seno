from typing import Any

from google.oauth2.credentials import Credentials

from app.integrations.google_auth import GOOGLE_SCOPES, GoogleAuth


SCOPES = GOOGLE_SCOPES


class GmailAuth:
    def __init__(
        self,
        client_secrets_file: str | None,
        token_file: str = "token.json",
        *,
        allow_interactive_oauth: bool = True,
    ) -> None:
        self.google_auth = GoogleAuth(
            client_secrets_file,
            token_file,
            SCOPES,
            allow_interactive_oauth=allow_interactive_oauth,
        )

    def get_credentials(self) -> Credentials:
        return self.google_auth.get_credentials()

    def build_service(self) -> Any:
        return self.google_auth.build_service("gmail", "v1")

    def _write_token(self, creds: Credentials) -> None:
        self.google_auth._write_token(creds)
