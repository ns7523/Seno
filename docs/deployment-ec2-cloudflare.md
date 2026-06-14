# EC2 + Cloudflare Tunnel Deployment Notes

Use the files in `deploy/systemd/` as starting points for production.

1. Copy the repo to `/opt/ai-email-agent`.
2. Create a Python 3.11 virtual environment at `/opt/ai-email-agent/venv`.
3. Install dependencies with `pip install -r requirements.txt`.
4. Copy `.env.production.example` to `.env.production` and fill secrets on the server only.
5. Store Gmail OAuth files under `/opt/ai-email-agent/secrets/`.
6. Install the FastAPI service:
   `sudo cp deploy/systemd/ai-email-agent.service /etc/systemd/system/`
7. Install Cloudflare tunnel config and service:
   `sudo cp deploy/systemd/cloudflared-email-agent.service /etc/systemd/system/`
8. Enable services:
   `sudo systemctl enable --now ai-email-agent cloudflared-email-agent`

Security notes:
- Keep Uvicorn bound to `127.0.0.1`; expose it only through Cloudflare Tunnel or a reverse proxy.
- Set `ADMIN_SECRET` and `TELEGRAM_WEBHOOK_SECRET`; startup fails in production without them.
- Protect `/status`, `/diagnostics/gmail`, and `/tasks/poll-once` with `X-ADMIN-SECRET`.
- Rotate Telegram/Gmail/LLM credentials if `.env`, `token.json`, or archives are exposed.
- Do not upload local `Archive.zip`, SQLite database files, `.env`, Gmail token files, or OAuth client secret files to GitHub, Render, or shared support channels. Treat those files as sensitive production material even when they are only local development copies.
