from github import Github
import os

g = Github(os.getenv('GITHUB_TOKEN'))

repo = g.get_repo('RayyanMinhaj/AI-PR-Review-Bot---Jenkins')

pr_number = int(os.getenv('PR_NUMBER'))
pull_request = repo.get_pull(pr_number)

disclaimer = """### Welcome to `SparklingCleanCode.com`, your automated AI PR Reviewing bot! 
Your report is being generated please wait...

> **DISCLAIMER: GPT3.5 TURBO only has the capability to process 16,385 tokens and output 4,096 tokens. If the Pull Request exceeds the limit by being too large, it will not be processed through AI and the results will be unreliable!**"""

pull_request.create_issue_comment(disclaimer)
