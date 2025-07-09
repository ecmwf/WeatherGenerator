# ruff: noqa: T201

#!/usr/bin/env -S uv run
# /// script
# dependencies = [ "BeautifulSoup4", "requests"
# ]
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///

"""
Checks that a pull request has a corresponding GitHub issue.

Source:
https://stackoverflow.com/questions/60717142/getting-linked-issues-and-projects-associated-with-a-pull-request-form-github-ap
"""

import re

import requests
from bs4 import BeautifulSoup

repo = "ecmwf/WeatherGenerator"

msg_template = """This pull request {pr} does not have a linked issue.
Please link it to an issue in the repository {repo} before merging.
The easiest way to do this is to add a comment with the issue number, like this:
Fixes #1234
This will automatically link the issue to the pull request.

If you just want to reference an issue without closing it, you can use:
Refs #1234

See https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check GitHub PR for linked issues.")
    parser.add_argument("pr", type=str, help="Pull request number")
    args = parser.parse_args()

    pr: str = args.pr
    pr = pr.split("/")[0]
    r = requests.get(f"https://github.com/{repo}/pull/{pr}")
    soup = BeautifulSoup(r.text, "html.parser")
    issueForm = soup.find_all("form", {"aria-label": re.compile("Link issues")})
    msg = msg_template.format(pr=pr, repo=repo)

    if not issueForm:
        print(msg)
        exit(1)
    issues = [i["href"] for i in issueForm[0].find_all("a")]
    issues = [i for i in issues if i is not None and repo in i]
    print(f"Linked issues for PR {pr}:")
    print(f"Found {len(issues)} linked issues.")
    print("\n".join(issues))
    if not issues:
        print(msg)
        exit(1)
