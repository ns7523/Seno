<img width="480" height="270" alt="image" src="https://github.com/user-attachments/assets/b8187f10-8ab4-4d9e-a490-0c000bba6e3e" />


# Sєno: Autonomous Executive Communication System

Production-ready FastAPI service that monitors Gmail, classifies incoming messages with Groq-first LLM routing plus local safety rules, auto-replies to safe messages, and sends risky messages to Telegram for approval.

## Architecture

- `app/gmail`: Gmail OAuth, unread email reader, threaded reply sender.
- `app/integrations`: shared Google OAuth plus Google Calendar scheduling services.
- `app/ai`: Groq-first classifier/provider routing, deterministic risk engine, local reply safety validation.
- `app/services`: Inbox orchestration, duplicate prevention, approval workflow.
- `app/telegram`: Telegram Bot API approval notifications and callback support.
- `app/memory`: sender history and trust scoring.
- `app/database.py`: SQLite persistence for emails, decisions, approvals, sender memory, and action logs.
- `app/main.py`: FastAPI app, health checks, Telegram webhook, APScheduler polling.

The first database backend is SQLite. The service layer only depends on the `Database` adapter, so it can later be swapped for PostgreSQL without changing Gmail, AI, or Telegram modules.

## Safety Model

The agent never auto-replies to noreply, newsletter, promotional, spam, or phishing-like messages. It requires approval for legal, financial, HR, angry, sensitive, attachment-bearing, low-confidence, unknown, or high-risk messages. Duplicate email IDs and duplicate approval callbacks are blocked in the database.

Groq is the default LLM provider for intent, urgency, tone, summary, confidence, and suggested reply generation. Deterministic local risk scoring and reply validation are always applied before and after model output, so no paid OpenAI API is required for safety or production operation. OpenAI remains an optional adapter only if explicitly configured.

## Local Setup

```bash
cd "/Users/143ns/BTECH/PORTFOLIO/EMail Automator"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with Groq, Gmail, and Telegram values. OpenAI is optional.

Run tests:

```bash
pytest -q
```

Run locally:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Or:

```bash
python -m app.main
```

Local development starts with `ENABLE_INBOX_MONITOR=false` by default, so `/health` works before Gmail, Groq, and Telegram credentials are configured. Set `ENABLE_INBOX_MONITOR=true` after `.env`, Gmail OAuth, Groq, and Telegram are ready.

Health check:

```bash
curl http://127.0.0.1:8000/health
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Production diagnostics:

```bash
curl https://<render-url>/
curl https://<render-url>/health
curl https://<render-url>/status
curl -H "X-Diagnostics-Secret: <DIAGNOSTICS_SECRET>" https://<render-url>/diagnostics/gmail
```

`/diagnostics/gmail` verifies the mounted Gmail client secret, token file, required scopes, Gmail profile access, and unread inbox query without returning email bodies or subjects.

## Google OAuth Setup

1. Go to Google Cloud Console.
2. Create a project and enable the Gmail API and Google Calendar API.
3. Configure OAuth consent screen.
4. Create an OAuth Client ID for a desktop app.
5. Download it as `client_secret.json`.
6. Generate the shared Gmail + Calendar token:

```bash
python scripts/generate_google_token.py
```

This creates `token.json` using the existing `client_secret.json`. The legacy command below is kept for compatibility and now requests the same shared scopes:

```bash
python scripts/generate_gmail_token.py
```

To intentionally force a fresh consent screen after adding Calendar scopes, run:

```bash
python scripts/generate_google_token.py --force
```

7. On Render, upload `client_secret.json` and `token.json` as secret files.

Required shared Google scopes:

- `https://www.googleapis.com/auth/gmail.modify`
- `https://www.googleapis.com/auth/gmail.send`
- `https://www.googleapis.com/auth/calendar`
- `https://www.googleapis.com/auth/calendar.events`

Enable Calendar features with:

```text
GOOGLE_CALENDAR_ENABLED=true
GOOGLE_CALENDAR_ID=primary
```

Seno uses the same token for Gmail and Calendar. If `token.json` is missing, invalid, expired without a refresh token, or missing Calendar scopes, the shared Google auth loader regenerates it from `client_secret.json`.

## Telegram Bot Setup

1. Message `@BotFather`.
2. Create a bot and copy the token into `TELEGRAM_BOT_TOKEN`.
3. Get your chat ID by messaging the bot, then visiting:

```text
https://api.telegram.org/bot<token>/getUpdates
```

4. Set `TELEGRAM_CHAT_ID`.
5. After deployment, configure the webhook:

```bash
curl "https://api.telegram.org/bot<token>/setWebhook?url=https://<render-url>/telegram/webhook&secret_token=<TELEGRAM_WEBHOOK_SECRET>"
```

## Groq Setup

Create a Groq API key and set:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
```

Optional OpenAI adapter:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

## Render Deployment

1. Push this repository to GitHub.
2. Create a new Render Blueprint from `render.yaml`.
3. Add secret environment variables:
   - `GROQ_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `DIAGNOSTICS_SECRET`
4. Add secret files:
   - `/etc/secrets/client_secret.json`
   - `/etc/secrets/token.json`
5. Deploy.
6. Confirm `https://<render-url>/health` returns `{"status":"ok"}`.
7. Set the Telegram webhook as shown above.

Render restarts the service automatically after crashes. APScheduler resumes polling on process startup. The SQLite disk at `/data` preserves processed message IDs, approvals, and memory.

## Docker

```bash
docker build -t autonomous-gmail-ai-agent .
docker run --env-file .env -p 8000:8000 autonomous-gmail-ai-agent
```

## Environment Variables

See `.env.example` for all supported variables. In production, these are required:

- `GROQ_API_KEY`
- `GMAIL_CLIENT_SECRETS_FILE`
- `GMAIL_TOKEN_FILE`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Troubleshooting

- Gmail or Calendar auth fails: regenerate the shared `token.json` locally with `python scripts/generate_google_token.py`, then redeploy the updated secret file.
- Telegram buttons do nothing: verify webhook URL, `TELEGRAM_WEBHOOK_SECRET`, and Render logs.
- Duplicate replies: check `emails` and `approvals` tables. Gmail message IDs are primary keys and approval rows are unique by Gmail ID.
- Groq outage or missing provider key: messages fall back to approval-required mode; local risk/safety rules still run.
- Render restarts: inspect action logs and service logs. The agent is designed to continue after restart without reprocessing stored Gmail IDs.

## Future Improvements

- PostgreSQL adapter with migrations.
- Gmail push notifications via Pub/Sub instead of polling.
- Admin UI for approvals and audit history.
- Per-sender custom reply style memory.
- Vector memory for long-running relationships.
