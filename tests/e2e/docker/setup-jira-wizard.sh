#!/usr/bin/env bash
# Automate Jira DC's first-run setup wizard over HTTP - no browser.
#
# Prereqs: `docker compose up -d jira-db jira` with the DB preconfigured
# via environment (this repo's docker-compose.yml does that), so the
# wizard skips its mode/database steps and starts at application
# properties.
#
# Usage:
#   bash setup-jira-wizard.sh [license-key]
#
# Without an argument the timebomb license is fetched automatically via
# get-jira-license.sh. Admin account created: admin / admin123 (the
# credentials the e2e suite expects). Verified against Jira 10.3.
set -euo pipefail

JIRA_URL="${JIRA_URL:-http://localhost:8080}"
ADMIN_USER="${ADMIN_USER:-admin}"
ADMIN_PASS="${ADMIN_PASS:-admin123}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LICENSE="${1:-$(bash "$SCRIPT_DIR/get-jira-license.sh")}"

CURL=(curl -fsS --noproxy '*')
JAR=$(mktemp)
trap 'rm -f "$JAR"' EXIT

echo "Waiting for Jira at $JIRA_URL ..."
for _ in $(seq 1 60); do
    if "${CURL[@]}" -o /dev/null "$JIRA_URL/status" 2>/dev/null; then
        break
    fi
    sleep 10
done

if "${CURL[@]}" -u "$ADMIN_USER:$ADMIN_PASS" "$JIRA_URL/rest/api/2/serverInfo" \
    -o /dev/null 2>/dev/null; then
    echo "Jira is already set up."
    exit 0
fi

token() {
    awk '/atlassian.xsrf.token/ {print $NF}' "$JAR" | tail -1
}

step() {
    local name=$1 path=$2
    shift 2
    echo "-> $name"
    # --data-urlencode makes this a POST; no -X POST, so a redirect
    # after the step is followed as GET instead of a forced POST (405)
    "${CURL[@]}" -b "$JAR" -c "$JAR" -o /dev/null -L \
        "$JIRA_URL/$path" \
        --data-urlencode "atl_token=$(token)" "$@"
}

# Prime session + XSRF cookie
"${CURL[@]}" -c "$JAR" -o /dev/null -L "$JIRA_URL/"

step "application properties" secure/SetupApplicationProperties.jspa \
    --data-urlencode "title=E2E Jira" \
    --data-urlencode "mode=private" \
    --data-urlencode "baseURL=$JIRA_URL" \
    --data-urlencode "nextStep=true"

step "license" secure/SetupLicense.jspa \
    --data-urlencode "setupLicenseKey=$LICENSE"

step "admin account" secure/SetupAdminAccount.jspa \
    --data-urlencode "fullname=Admin" \
    --data-urlencode "email=admin@example.com" \
    --data-urlencode "username=$ADMIN_USER" \
    --data-urlencode "password=$ADMIN_PASS" \
    --data-urlencode "confirm=$ADMIN_PASS"

step "mail notifications" secure/SetupMailNotifications.jspa \
    --data-urlencode "noemail=true"

# Jira locks an account behind a CAPTCHA after three failed logins, and
# REST refuses basic auth for a few minutes after the wizard while the
# instance finishes starting. Every probe in that window counts as a
# failure, so the account is locked before it is ever usable - and the
# CAPTCHA can only be cleared through the web UI. Raising the threshold
# makes the disposable test instance immune; never do this on a real one.
echo "-> disabling the CAPTCHA lockout (test instance only)"
JIRA_CONTAINER="${JIRA_CONTAINER:-docker-jira-1}"
JIRA_HOME_DIR="/var/atlassian/application-data/jira"
if docker exec "$JIRA_CONTAINER" sh -c \
    "printf 'jira.maximum.authentication.attempts.allowed = 1000000\\n' \
        >> $JIRA_HOME_DIR/jira-config.properties" 2>/dev/null; then
    docker restart "$JIRA_CONTAINER" >/dev/null
else
    echo "   (skipped: container '$JIRA_CONTAINER' not reachable)"
fi

# Wait for REST to answer at all before authenticating: a 503 during
# startup would otherwise be counted as a failed login.
echo "Waiting for the REST API ..."
for _ in $(seq 1 60); do
    code=$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' \
        "$JIRA_URL/rest/api/2/serverInfo" || true)
    [ "$code" != "503" ] && [ "$code" != "000" ] && break
    sleep 10
done
sleep 30

echo "Verifying ..."
"${CURL[@]}" -u "$ADMIN_USER:$ADMIN_PASS" "$JIRA_URL/rest/api/2/serverInfo" \
    | head -c 200
echo
echo "Done. Continue with: bash setup-test-data.sh"
