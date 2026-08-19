#!/usr/bin/env bash
# Fetch Atlassian's public Jira Software Data Center timebomb license
# (10 users, valid 3 hours from application - free, no account needed).
#
# Prints the license key on stdout. Use it for the Jira setup wizard,
# either manually or via setup-jira-wizard.sh (which calls this script
# itself when no key is provided).
#
# Source page (also lists Confluence DC and other products):
#   https://developer.atlassian.com/platform/marketplace/timebomb-licenses-for-testing-server-apps/
set -euo pipefail

PAGE_URL="https://developer.atlassian.com/platform/marketplace/timebomb-licenses-for-testing-server-apps/"
PRODUCT="${1:-Jira Software Data Center}"

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
curl -fsSL "$PAGE_URL" -o "$tmp"

python3 - "$PRODUCT" "$tmp" <<'PYEOF'
import re
import sys

product, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8", errors="replace") as f:
    html = f.read()
# The page embeds its markdown in a JS bundle with literal \n sequences:
#   **<product> license, expires in 3 hours**\n\n``` bash\n<KEY>\n```
pattern = re.compile(
    re.escape(product)
    + r" license, expires in 3 hours.*?```[ ]?bash\\n(.*?)\\n```",
    re.S,
)
match = pattern.search(html)
if not match:
    sys.exit(
        f"Could not find the '{product}' timebomb key - the page layout "
        "may have changed. Copy it manually from the page."
    )
print(match.group(1).replace("\\n", "").strip())
PYEOF
