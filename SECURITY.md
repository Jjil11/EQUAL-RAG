# Security and disclosure

Never commit access tokens, SSH passwords/keys, `.env` secrets, private machine addresses, model weights, or unreviewed logs. Use GitHub credential storage or a suitable external secret manager; do not place credentials in repository URLs or source code.

This snapshot intentionally excludes machine orchestration/SSH helpers, raw predictions and retrieved text. The release verifier checks for common credential and private-path patterns, but no pattern scanner guarantees absence of every possible secret.

Report security concerns privately to repository owner `Jjil11` through an agreed private channel. Do not put a live secret into a GitHub issue. If a secret is accidentally published, revoke it; removing the current file alone does not remove it from Git history.
