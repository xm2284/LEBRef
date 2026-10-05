"""将 GitHub 仓库的 description 与 topics 同步到 Gitee。

在 GitHub Actions 中运行，依赖以下环境变量：
  GH_REPO          GitHub 仓库全名（owner/repo）
  GH_TOKEN         GitHub 令牌（read 即可，用于读取 description/topics）
  REPO_NAME        仓库名
  GITEE_USERNAME   Gitee 用户名
  GITEE_TOKEN      Gitee 令牌

只做两件事：
  1. PATCH Gitee 仓库 description
  2. PUT   Gitee 仓库 project_labels（用 GitHub topics 覆盖）
失败只记录，不中断工作流。
"""

import json
import os
import urllib.error
import urllib.request


def _get(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _send(method: str, url: str, data: dict) -> tuple[int, str]:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main() -> None:
    gh_repo = os.environ["GH_REPO"]
    repo = os.environ["REPO_NAME"]
    user = os.environ["GITEE_USERNAME"]
    token = os.environ["GITEE_TOKEN"]

    gh_headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
        "User-Agent": "gitee-mirror",
    }
    meta = _get(f"https://api.github.com/repos/{gh_repo}", gh_headers)
    description = meta.get("description") or ""
    topics = meta.get("topics") or []

    print(f"description: {description}")
    print(f"topics: {topics}")

    base = f"https://gitee.com/api/v5/repos/{user}/{repo}"

    status, text = _send("PATCH", base, {"access_token": token, "description": description})
    print(f"PATCH description -> {status}: {text[:200]}")

    status, text = _send("PUT", f"{base}/project_labels", {"access_token": token, "labels": topics})
    print(f"PUT project_labels -> {status}: {text[:300]}")


if __name__ == "__main__":
    main()
