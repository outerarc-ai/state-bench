# Security policy

state-bench reads and writes local trace files and, in future releases, will send
requests to inference servers you point it at. Traces are designed to exclude prompt and
output content; do not add fields that carry it.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on the
[repository Security page](https://github.com/outerarc-ai/state-bench/security) when
available. Do not include exploitable details, credentials, or private data in a public
issue. If private reporting is unavailable, contact the maintainers through their GitHub
profiles to arrange a private channel.

Include the affected version or commit, a concise reproduction, impact, and any suggested
mitigation. Maintainers will acknowledge receipt and coordinate disclosure before
publishing a fix.

If you find that a published trace exposes content it should not, report it the same way
so it can be removed.
