# Setup — credentials and connection

Nothing here is needed to run the dry run. This is the checklist for when you're ready to connect the real accounts.

Each item says what it is, why the pipeline needs it, and what it can do — so you can decide how much access you're comfortable granting before creating anything.

---

## 0. Decide the Drive authentication mode first

**This is the decision that shapes everything else.** Answer one question: is the Drive containing `5. AS Chemistry` on a **Google Workspace** account or a **personal Gmail** account?

### Why it matters

A Google service account is its own identity with **no Drive storage quota**. It can read and reorganise files in a folder shared with it, but the moment it tries to *own* a new file in someone's My Drive, the upload fails with a storage-quota error — which reads as nonsense, because the human's Drive has plenty of space. This trips up nearly every first attempt at this integration.

### The three modes

| Your situation | `GOOGLE_AUTH_MODE` | How it works |
|---|---|---|
| **Workspace, content on a Shared Drive** | `service_account` | Files are owned by the Shared Drive, not by any user. The service account is added as a member. **Cleanest option — recommended.** |
| **Workspace, content in a user's My Drive** | `impersonation` | Domain-wide delegation: the service account acts as a real user. Needs Workspace admin to authorise it. |
| **Personal Gmail** | `oauth_user` | Service accounts won't work. You authorise once through a browser and the pipeline stores a refresh token. |

If you're on a personal account, **consider moving this content to a Shared Drive on a Workspace account before connecting anything.** It's a one-time move that eliminates a whole class of ongoing problems — quota errors, ownership confusion, and tokens that need periodic re-consent.

---

## 1. Google Cloud project and Drive API

1. Create a project at <https://console.cloud.google.com>.
2. Enable the **Google Drive API**.
3. Create credentials per your mode below.

### Scope

The pipeline requests `https://www.googleapis.com/auth/drive` — full Drive access.

This is broader than ideal and it's worth understanding why it's unavoidable: the narrower `drive.file` scope only grants access to files the application itself created. Your recordings and folders were created by people, so `drive.file` cannot see them at all. Reorganising a pre-existing tree requires the full scope.

What the pipeline does with it, enforced in code ([ADR-009](DECISIONS.md#adr-009--full-drive-scope-tightly-constrained)):

- Operates only under `GOOGLE_DRIVE_ROOT_FOLDER_ID`; anything resolving outside is refused before the API call
- **Never deletes** — there is no delete call in the client
- Logs every mutation with before/after state, so anything can be reversed
- Supports `--dry-run`, which logs intended changes without making them

### Mode: `service_account` (Workspace + Shared Drive)

1. IAM & Admin → Service Accounts → create one.
2. Create a JSON key. Save it to `credentials/service-account.json` — **gitignored, never commit it**.
3. Open the Shared Drive → Manage members → add the service account's email (`...@....iam.gserviceaccount.com`) as **Content manager**.
4. Set `GOOGLE_SHARED_DRIVE_ID` as well as the folder IDs.

### Mode: `impersonation` (Workspace + My Drive)

1. Create the service account and key as above.
2. Enable **domain-wide delegation** on it; note the client ID.
3. Workspace Admin console → Security → API controls → Domain-wide delegation → authorise that client ID for the `drive` scope.
4. Set `GOOGLE_IMPERSONATE_SUBJECT` to the email of the user who owns the folders.

### Mode: `oauth_user` (personal Gmail)

1. Configure the OAuth consent screen (External; add your own account as a test user).
2. Create an **OAuth client ID** of type *Desktop app*.
3. Run `python scripts/google_oauth_setup.py` — it opens a browser, you consent once, and it prints a refresh token.
4. Put the client ID, secret, and refresh token in `.env`.

Refresh tokens on apps still in "Testing" status expire after about a week. Publish the consent screen to stop that.

### Folder IDs

From a Drive URL — `https://drive.google.com/drive/folders/1A2B3C...` — the ID is the trailing segment.

| Variable | Which folder |
|---|---|
| `GOOGLE_DRIVE_ROOT_FOLDER_ID` | The managed root, e.g. the parent of `5. AS Chemistry` |
| `GOOGLE_DRIVE_INBOX_FOLDER_ID` | Where new recordings land before processing |
| `GOOGLE_DRIVE_REVIEW_FOLDER_ID` | `_Needs Review` — create it if it doesn't exist |

---

## 2. Zoom

A **Server-to-Server OAuth** app. (JWT apps were retired in 2023 — if a guide mentions JWT, it's out of date.)

1. <https://marketplace.zoom.us> → Develop → Build App → **Server-to-Server OAuth**.
2. Note the **Account ID**, **Client ID**, **Client Secret**.
3. Scopes — grant only these two:
   - `recording:read:admin` — list and download cloud recordings
   - `user:read:admin` — resolve host email to a teacher name
4. **Feature → Event Subscriptions**:
   - Event notification endpoint: `https://YOUR_DOMAIN/webhooks/zoom`
   - Subscribe to **Recording → All Recordings have completed**
   - Copy the **Secret Token** into `ZOOM_WEBHOOK_SECRET_TOKEN`
5. Click **Validate**. Zoom sends a challenge that the running API service answers automatically — so start the service before validating.
6. Activate the app.

### The endpoint must be publicly reachable

Zoom needs to POST to you over HTTPS. For development, a tunnel (`ngrok`, `cloudflared`) is fine — update the endpoint URL in the Zoom app when the tunnel URL changes. For production, use a stable domain with a real certificate.

**Both signature and timestamp are verified before any processing.** An unverified webhook endpoint lets anyone enqueue work.

---

## 3. Anthropic API key

For classification.

1. <https://console.anthropic.com> → API keys → create one.
2. `ANTHROPIC_API_KEY=sk-ant-...` in `.env`.

Cost is a few cents per lecture. Set a spend limit in the console for peace of mind.

---

## 4. Infrastructure

| Component | Dev | Production |
|---|---|---|
| Database | SQLite (default, no setup) | Postgres — change `DATABASE_URL` and run migrations |
| Queue | Redis | Redis (managed is fine) |
| ffmpeg | Only for the Zoom path — install and put on PATH | Included in the worker container image |
| GPU | Optional | Needed for self-hosted ASR; skip it and use a hosted ASR provider instead |

Install extras by role rather than all at once:

```bash
pip install -e ".[dev]"                    # tests + dry run; no ML deps, seconds to install
pip install -e ".[asr,syllabus]"           # transcription + syllabus ingestion; no GPU needed
pip install -e ".[dev,asr,embeddings]"     # full transcription worker
```

`[embeddings]` (semantic retrieval) is optional everywhere — without it retrieval falls back to lexical-only, which is worse at paraphrase but fully functional.

`ffmpeg` is currently **not installed** on this machine. Windows: `winget install Gyan.FFmpeg`, then restart the shell. It is needed to extract audio from a **Zoom** download; it is *not* needed to transcribe a local video, because faster-whisper decodes through PyAV, which carries its own ffmpeg libraries. `scripts/rename_folder.py` therefore runs on this machine as it stands.

**Python versions.** Everything runs on 3.11+, including the transcription worker — verified on this machine's 3.14.3. The `asr` extra used to be pinned below 3.14 because `ctranslate2` wheels lagged new CPython releases; that is no longer true (`ctranslate2` 4.8 and `faster-whisper` 1.2 both publish cp314 wheels), and the marker has been removed. `verify_config.py` now checks by importing rather than by comparing version numbers, so it reports what is actually installed.

---

## 5. Connect and verify, in this order

Do not skip to step 5. Each step is verifiable on its own, and connecting everything at once makes a failure hard to localise.

```bash
# 1. Config only — no external calls. Confirms .env parses and IDs are present.
python scripts/verify_config.py

# 2. Read-only Drive. Lists the tree and resolves syllabus nodes to folder IDs.
#    Confirms auth works and folder names match the taxonomy.
python scripts/bootstrap_drive_map.py --dry-run

# 3. Zoom read-only. Lists recent recordings without downloading.
python scripts/verify_zoom.py

# 4. Full pipeline, no writes. Transcribes and classifies one real recording
#    and reports what it WOULD do.
python scripts/process_one.py --recording-id <id> --dry-run

# 5. Live, one file.
python scripts/process_one.py --recording-id <id>
```

Then start the services and let the webhook drive it.

---

## Security checklist

- [ ] `.env` and `credentials/` are gitignored — verify with `git status` before your first commit
- [ ] Service account key stored with restrictive file permissions
- [ ] Zoom app scopes limited to the two listed above
- [ ] Anthropic key has a spend limit set
- [ ] `GOOGLE_DRIVE_ROOT_FOLDER_ID` is as narrow as possible — the pipeline cannot touch anything outside it
- [ ] Production secrets in a secret manager, not a `.env` file on disk
- [ ] A retention and access policy decided for transcripts — they contain student voices and names, and are more easily read than the video they came from

---

## Nothing here is needed yet

The dry run (`python scripts/dry_run.py`) exercises the full pipeline against a local fake Drive with no credentials at all. Use it to review behaviour before granting any access.
