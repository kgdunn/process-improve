# Claude Skills

Skills that let someone use the `process-improve` tooling from their own
Claude account, without a server, an account on anyone's website, or an API
key held by a third party.

The library is the computation; the skill carries the methodology. That split
is deliberate: a language model asked to *do* design of experiments will
happily write out a design matrix that looks right and is not (see
`doe-designer/references/verification.md` for the evidence). The skill's job
is to route the model to a catalogue-backed generator and a verifier instead.

## Available skills

| Skill | What it does |
|---|---|
| [`doe-designer`](doe-designer/) | Plan, generate, verify and analyse designed experiments |

## Installing

### Claude Code, as a plugin

```
/plugin marketplace add kgdunn/process-improve
/plugin install doe-designer@process-improve
```

### Claude Code or Claude Desktop, as a local skill

Copy the skill folder into either location:

```bash
# Available in every project
cp -r skills/doe-designer ~/.claude/skills/

# Available in one project only
cp -r skills/doe-designer .claude/skills/
```

### claude.ai

Zip the skill folder and upload it under Settings, Capabilities, Skills.

```bash
cd skills && zip -r doe-designer.zip doe-designer
```

Note that the claude.ai code sandbox has restricted network access, so
`process-improve` may not be installable at runtime there. The Claude Code and
Claude Desktop routes are the reliable ones.

## The MCP alternative

If you want the tools without the methodology, `process-improve` also ships an
MCP server exposing the same registry:

```bash
pip install 'process-improve[mcp]'
```

```json
{
  "mcpServers": {
    "process-improve": {
      "command": "process-improve-mcp"
    }
  }
}
```

The two compose well. The MCP server gives Claude the tools; the skill tells
it when and why to reach for each one, and what to do with the answer. Running
both is the best of it.

## Requirements

```bash
pip install 'process-improve[expt,plotting]'
```

Each script also carries a PEP 723 header, so `uv run --script <script>` will
resolve its own dependencies with nothing installed in advance.

## Licence

MIT, the same as the rest of the repository. The skill text and code are
covered; the output is not engineering or statistical validation. Review any
design before spending lab time on it, which is exactly what the verification
step in `doe-designer` is for.
