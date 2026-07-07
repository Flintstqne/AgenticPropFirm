# Version Control Strategy

This document holds the full version control policy for this repository. AGENTS.md and CLAUDE.md link here rather than repeating this content, so keep this file as the single source for branching rules and commit rules.

## Branch Structure

main: the production branch. What the live system runs. Protected. No direct commits, ever.

staging: the pre-production branch. Mirrors production. The last check before a release reaches main.

dev: the active development branch. Every finished feature lands here first, before staging, before main.

feature branches: one branch per feature or fix, branched from dev, named `feature/<short-description>`.

## Flow

Branch off dev for every feature or fix.

```
git checkout dev
git pull
git checkout -b feature/<short-description>
```

Open a pull request from the feature branch into dev once the feature works and passes its own tests.

Review the full diff before merging, even working solo. Reading the diff inside the pull request stands in as the review step. Do not skip this step because nobody else works on the project.

Merge dev into staging on a regular cadence, or once a group of finished features reaches a stable point worth testing together.

Test staging the way you would test production, since staging mirrors production exactly.

Open a pull request from staging into main once staging holds a version ready to ship.

Merge into main only through that pull request. Never through a direct push, regardless of how small the change looks.

## Branch Protection

Turn on these settings for main and for staging inside the repository settings on GitHub:

Require a pull request before merging.

Require the branch to sit up to date with its target before merging.

Restrict direct pushes, so nobody, including you, pushes straight to either branch.

Working solo does not remove the value of these settings. A required pull request forces a diff review before a change lands, and a bad merge on main becomes a one click rollback instead of a scramble at 2am.

## Direct Push Rule

Never push directly to main. Never push directly to staging. Every change moves through a feature branch, a pull request, and a merge, every time, for every change, no matter how small. A one line fix follows the same path as a full feature.

## Commit Authorship

Every commit on this repository carries your own git identity, the name and email configured in your own global git config.

An agent working inside this repository never commits under its own name, and never sets `user.name` or `user.email` to an agent identity, a bot account, or any identity other than yours.

An agent may stage changes and draft a commit message. The agent stops there. You review the staged diff and run `git commit` and `git push` yourself, from your own terminal, under your own git identity. An agent never runs `git commit` or `git push` on this repository on its own initiative, regardless of how confident the agent feels about the change.

If an agent operates inside a terminal session already authenticated under your git identity, every commit made through that session still needs your direct action to commit and push. The agent preparing a change is not the same action as you approving and shipping that change.
