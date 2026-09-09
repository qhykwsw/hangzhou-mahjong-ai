#!/usr/bin/env python3
"""Refresh the complete public platform reference bundle; no authenticated calls."""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import ssl
import tempfile
import urllib.request

from mahjong_ai.platform.calibration import calibrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://10.240.169.190:18080")
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    context = (
        ssl._create_unverified_context()
        if args.insecure
        else ssl.create_default_context()
    )
    target = Path(__file__).resolve().parents[1] / "docs/sources"

    def fetch(path, body=None):
        request = urllib.request.Request(
            base + path,
            data=None
            if body is None
            else json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, context=context, timeout=30) as response:
            return response.read(), response.status

    def get_json(path):
        return json.loads(fetch(path)[0])

    version = get_json("/portal/api/guide/version")
    guide = get_json("/portal/api/guide")
    if guide["version"] != version["version"] or guide["format"] != "html":
        raise ValueError("guide and version metadata disagree")
    portal = fetch("/portal/")[0].decode()
    start = re.search(r'<section\b[^>]*\bid="view-flow"[^>]*>', portal)
    if start is None:
        raise ValueError("portal flow section missing")
    depth = 0
    flow = None
    for tag in re.finditer(r"</?section\b[^>]*>", portal[start.start() :]):
        depth += -1 if tag.group().startswith("</") else 1
        if depth == 0:
            flow = portal[start.start() : start.start() + tag.end()]
            break
    if flow is None or "/portal/replay.js" not in portal:
        raise ValueError(
            "portal flow or replay reference changed; inspect current portal"
        )
    replay = fetch("/portal/replay.js")[0]
    if len(replay) < 100 or replay.lstrip().startswith(b"<"):
        raise ValueError("unexpected replay response")
    blocks = [
        html.unescape(x)
        for x in re.findall(
            r"<pre\b[^>]*>\s*<code\b[^>]*>(.*?)</code>\s*</pre>", guide["content"], re.S
        )
    ]
    bots = [b for b in blocks if b.startswith("#!/usr/bin/env python3")]
    if len(bots) != 1:
        raise ValueError("expected exactly one official Python Bot example")
    compile(bots[0], "official-minimal-bot.py", "exec")

    # Fetch and validate everything before replacing any current reference.
    with tempfile.TemporaryDirectory(prefix="platform-sources-") as temporary:
        staging = Path(temporary)
        calibration = calibrate(
            base, staging / "fan-calibration.json", insecure=args.insecure
        )
        request = calibration["records"][0]["request"]
        response, status = fetch("/portal/api/tools/fan-calc", request)
        acquired = datetime.now(timezone.utc).isoformat()
        four_whites = {
            "source": base + "/portal/api/tools/fan-calc",
            "retrieved_at": acquired,
            "guide_version": version["version"],
            "method": "POST",
            "endpoint_semantics": "pure stateless calculation",
            "request": request,
            "status": status,
            "response": json.loads(response),
        }
        if (
            get_json("/portal/api/guide/version") != version
            or calibration["guide_version"] != version["version"]
        ):
            raise ValueError(
                "platform version changed during refresh; retry the complete bundle"
            )
        calibration["retrieved_at"] = acquired
        payloads = {
            "platform-guide.html": guide["content"].encode(),
            "platform-flow.html": flow.encode(),
            "platform-replay.js": replay,
            "official-minimal-bot.py": bots[0].encode(),
        }
        for name, data in {
            "platform-guide-version.json": version,
            "fan-calibration.json": calibration,
            "fan-calc-four-whites.json": four_whites,
        }.items():
            payloads[name] = (
                json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            ).encode()
        for name, data in payloads.items():
            (target / name).write_bytes(data)
        print(
            json.dumps(
                {
                    "guide_version": version["version"],
                    "updated": list(payloads),
                    "calibration_matched": calibration["matched"],
                    "calibration_samples": calibration["samples"],
                },
                ensure_ascii=False,
            )
        )
        if calibration["matched"] != calibration["samples"]:
            raise SystemExit(
                "Fresh evidence saved; local fan calculation differs from the platform. Inspect records."
            )


if __name__ == "__main__":
    main()
