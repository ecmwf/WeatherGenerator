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

import requests
from bs4 import BeautifulSoup
import re

repo = "ecmwf/WeatherGenerator"

msg_template = """"This pull request {pr} does not have a linked issue.
Please link it to an issue in the repository {repo} before merging.
The easiest way to do this is to add a comment with the issue number, like this:
Fixes #1234
This will automatically link the issue to the pull request."""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check GitHub PR for linked issues.")
    parser.add_argument("pr", type=str, help="Pull request number")
    args = parser.parse_args()
    
    pr = args.pr
    r = requests.get(f"https://github.com/{repo}/pull/{pr}")
    soup = BeautifulSoup(r.text, 'html.parser')
    issueForm = soup.find_all("form", { "aria-label": re.compile('Link issues')})
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
        print(f"No linked issues found for PR {pr}.")
        exit(1)

