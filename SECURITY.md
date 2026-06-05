# Security Policy

`process-improve` is a data-analysis library that can also expose its analysis
registry as agent-callable tools over a
[Model Context Protocol (MCP)](https://modelcontextprotocol.io) server
(`process_improve.mcp_server`). Because that surface can be driven by an LLM or,
when deliberately fronted by HTTP, by remote callers, the project takes security
reports seriously.

## Supported versions

Only the latest released version on
[PyPI](https://pypi.org/project/process-improve/) receives security fixes. There
are no long-term-support branches; please upgrade to the most recent release
before reporting an issue, and pin a minimum version once a fix ships.

| Version          | Supported          |
| ---------------- | ------------------ |
| Latest release   | :white_check_mark: |
| Older releases   | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue, pull request, or discussion for a
security vulnerability.** Public disclosure before a fix is available puts every
user at risk.

Instead, use one of these private channels:

1. **GitHub private vulnerability reporting (preferred).** Go to the
   [Security tab](https://github.com/kgdunn/process-improve/security/advisories)
   and click **"Report a vulnerability"**. This opens a private advisory visible
   only to you and the maintainers.
2. **Email.** Write to the maintainer, Kevin Dunn, at
   `kgdunn@gmail.com` with a subject line starting `[process-improve security]`.

Please include, as far as you can:

- the affected version (`python -c "import process_improve; print(process_improve.__name__)"`
  plus the installed version from `pip show process-improve`),
- a description of the vulnerability and its impact,
- a minimal reproduction (a short script, the tool call, or the input payload),
- and whether you believe it is reachable under the **untrusted** threat model
  (MCP server reachable by hostile callers) or only the **local-trusted** model
  (the server only ever drives the owner's own LLM on the owner's machine). See
  [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) for how these two models are scored.

## What to expect

- **Acknowledgement** within 7 days that the report was received.
- **An initial assessment** (severity, threat model, whether it reproduces)
  within 14 days.
- **A fix or mitigation plan** communicated before any public disclosure. Fixes
  ship as a new PyPI release with an entry in
  [`CHANGELOG.md`](CHANGELOG.md) and, where applicable, a published
  [GitHub Security Advisory](https://github.com/kgdunn/process-improve/security/advisories).
- **Credit** for the reporter in the advisory and changelog, unless you ask to
  remain anonymous.

We ask that you give us a reasonable opportunity to release a fix before any
public disclosure (coordinated disclosure).

## Scope and threat model

The project's own catalogue of past findings, the threat models, and the
hardening already in place live in [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md).
In summary:

- Tool inputs are validated against per-tool pydantic models
  (`extra="forbid"`), and the safe-execution path
  (`PROCESS_IMPROVE_MCP_SAFE_MODE=1`) adds input-size caps, a wall-clock
  timeout with worker termination, and a per-subprocess memory cap.
- Model formulas are validated against a strict Wilkinson-subset allowlist
  before they reach `patsy`/`statsmodels`, which would otherwise evaluate them
  as arbitrary Python.

Reports that demonstrate a bypass of any of these controls, or a new
code-execution / information-disclosure / denial-of-service vector, are
especially valuable.

## Out of scope

- Vulnerabilities in third-party dependencies (report those upstream; we will
  bump the pin once a fixed version is available).
- Denial of service that requires the **local-trusted** model only (a user
  running the stdio server on their own machine can already run arbitrary code).
- Findings that require a non-default, explicitly unsafe configuration that the
  documentation warns against.
