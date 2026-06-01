# Intraday snapshot — external pinger setup

GitHub's built-in `schedule` cron for `.github/workflows/intraday.yml` is
unreliable for this repo: across a full trading day it fired **zero**
times, while manual / API `workflow_dispatch` triggers succeed every time
(see PLAN.md 2026-06-01). The fix is to drive the snapshot from a
**reliable external timer** that calls the GitHub `workflow_dispatch` API.
No Mac needs to be on.

The in-repo cron is kept as a harmless free backup — if GitHub ever fires
it, the market-closed freeze prevents duplicate commits.

---

## What the pinger does

A single HTTP POST, on a timer:

```
POST https://api.github.com/repos/carlchen4/stock-picker-claude/actions/workflows/intraday.yml/dispatches
Headers:
  Accept: application/vnd.github+json
  Authorization: Bearer <TOKEN>
  X-GitHub-Api-Version: 2022-11-28
  Content-Type: application/json
Body:
  {"ref":"main"}
Expected response: 204 No Content
```

The workflow self-gates: `intraday_monitor.py --once` only rewrites the
page when a ticker's latest bar is **today (ET)**, so the pinger can fire
on a simple "every 5 min" schedule 24/7 — off-hours pings produce no
commit churn. (Restricting the pinger to market hours / weekdays is
optional tidiness, not required.)

---

## Step 1 — create a scoped token (GitHub, ~2 min)

Use a **fine-grained PAT** limited to this one repo, minimum scope:

1. https://github.com/settings/personal-access-tokens/new
2. **Resource owner:** carlchen4 · **Repository access:** Only select
   repositories → `stock-picker-claude`
3. **Permissions → Repository → Actions:** **Read and write**
   (that single permission is enough to dispatch a workflow)
4. **Expiration:** set one (e.g. 90 days) and calendar a renewal
5. Generate, copy the `github_pat_…` value (shown once)

> A classic PAT with the `workflow` scope also works, but it can touch
> every repo you own — prefer the fine-grained, single-repo token above.

Blast radius if this token leaks: an attacker could only dispatch
workflows on this one **public** repo. Low, but rotate on expiry.

## Step 2 — create the cron job (cron-job.org free tier, ~3 min)

1. Sign up at https://cron-job.org → **Create cronjob**
2. **URL:** `https://api.github.com/repos/carlchen4/stock-picker-claude/actions/workflows/intraday.yml/dispatches`
3. **Schedule:** every 5 minutes (free tier supports down to 1 min)
4. **Advanced → Request method:** `POST`
5. **Advanced → Headers** (add each):
   - `Accept: application/vnd.github+json`
   - `Authorization: Bearer github_pat_…`  ← your token
   - `X-GitHub-Api-Version: 2022-11-28`
   - `Content-Type: application/json`
6. **Advanced → Request body:** `{"ref":"main"}`
7. Save. cron-job.org treats HTTP 2xx as success — expect **204**.

Any equivalent free scheduler works (UptimeRobot "keyword"/POST monitors,
Pipedream, Val Town, GitHub Actions on a *different* always-active repo,
etc.) — same URL / headers / body.

---

## Verify

From a shell with the token exported:

```sh
TOKEN=github_pat_xxx
curl -sS -o /dev/null -w '%{http_code}\n' \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/carlchen4/stock-picker-claude/actions/workflows/intraday.yml/dispatches \
  -d '{"ref":"main"}'
# → 204
```

Or, if `gh` is logged in (`gh auth status`):

```sh
gh api --method POST \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  /repos/carlchen4/stock-picker-claude/actions/workflows/intraday.yml/dispatches \
  -f ref=main
```

Then confirm a run appeared:

```sh
gh run list -R carlchen4/stock-picker-claude --workflow intraday.yml -L 3
```

The dashboard updates at: https://carlchen4.github.io/stock-picker-claude/intraday.html
