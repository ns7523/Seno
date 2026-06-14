from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.utils.logger import get_logger


logger = get_logger(__name__)


GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]

CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
]

GOOGLE_SCOPES = [*GMAIL_SCOPES, *CALENDAR_SCOPES]


class GoogleAuth:
    """Shared OAuth loader for Gmail and Calendar.

    Gmail and Calendar intentionally use one token file. If the stored token is
    missing Calendar scopes, expired without a refresh token, or otherwise
    invalid, this class falls back to the installed-app OAuth flow and rewrites
    token.json with the complete shared scope set.
    """

    def __init__(
        self,
        client_secrets_file: str | None,
        token_file: str = "token.json",
        scopes: list[str] | None = None,
        *,
        allow_interactive_oauth: bool = True,
    ) -> None:
        self.client_secrets_file = client_secrets_file
        self.token_file = Path(token_file)
        self.scopes = scopes or GOOGLE_SCOPES
        self.allow_interactive_oauth = allow_interactive_oauth

    def get_credentials(self) -> Credentials:
        creds = self._load_token()
        if creds and self._credentials_cover_scopes(creds) and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self._write_token(creds)
            except RefreshError as exc:
                logger.warning("Google token refresh failed; OAuth regeneration required", extra={"error": str(exc)})
                creds = None
        if not creds or not creds.valid or not self._credentials_cover_scopes(creds):
            creds = self._run_oauth_flow()
            self._write_token(creds)
        return creds

    def regenerate_credentials(self) -> Credentials:
        creds = self._run_oauth_flow()
        self._write_token(creds)
        return creds

    def build_service(self, service_name: str, version: str) -> Any:
        return build(service_name, version, credentials=self.get_credentials(), cache_discovery=False)

    def _load_token(self) -> Credentials | None:
        if not self.token_file.exists():
            logger.info("Google OAuth token missing; generation required", extra={"token_file": str(self.token_file)})
            return None
        if not self._token_file_covers_scopes():
            logger.info("Google OAuth token is missing required scopes; regeneration required", extra={"token_file": str(self.token_file)})
            return None
        try:
            return Credentials.from_authorized_user_file(str(self.token_file), self.scopes)
        except (ValueError, OSError) as exc:
            logger.warning("Google OAuth token could not be loaded; regeneration required", extra={"token_file": str(self.token_file), "error": str(exc)})
            return None

    def _run_oauth_flow(self) -> Credentials:
        if not self.allow_interactive_oauth:
            raise RuntimeError(
                "Interactive Google OAuth is disabled for this runtime. "
                "Regenerate token.json locally with scripts/generate_google_token.py."
            )
        if not self.client_secrets_file:
            raise RuntimeError("GMAIL_CLIENT_SECRETS_FILE is required to create shared Google credentials")
        if not Path(self.client_secrets_file).exists():
            raise RuntimeError(f"Google OAuth client secret file not found: {self.client_secrets_file}")
        logger.info("Starting shared Google OAuth flow", extra={"client_secrets_file": self.client_secrets_file, "scopes": self.scopes})
        flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets_file, self.scopes)
        return flow.run_local_server(
            port=0,
            authorization_prompt_message=(
                "\nOpen this URL in your browser, approve Gmail and Calendar access, "
                "then return here:\n\n{url}\n\n"
            ),
            success_message="Google OAuth complete. You can close this browser tab.",
        )

    def _write_token(self, creds: Credentials) -> None:
        try:
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            self.token_file.write_text(creds.to_json(), encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Google token refresh/generation succeeded but token file could not be updated",
                extra={"token_file": str(self.token_file), "error": str(exc)},
            )

    def _credentials_cover_scopes(self, creds: Credentials) -> bool:
        if hasattr(creds, "has_scopes"):
            try:
                return bool(creds.has_scopes(self.scopes))
            except TypeError:
                pass
        granted = set(getattr(creds, "scopes", None) or getattr(creds, "_scopes", None) or [])
        return set(self.scopes).issubset(granted) if granted else True

    def _token_file_covers_scopes(self) -> bool:
        try:
            payload = json.loads(self.token_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True
        stored_scopes = payload.get("scopes") or payload.get("granted_scopes")
        if isinstance(stored_scopes, str):
            stored_scopes = stored_scopes.split()
        if not stored_scopes:
            return True
        return set(self.scopes).issubset(set(stored_scopes))


def get_google_credentials(client_secrets_file: str | None, token_file: str = "token.json", scopes: list[str] | None = None) -> Credentials:
    return GoogleAuth(client_secrets_file, token_file, scopes).get_credentials()


def build_google_service(client_secrets_file: str | None, token_file: str, service_name: str, version: str) -> Any:
    return GoogleAuth(client_secrets_file, token_file).build_service(service_name, version)
