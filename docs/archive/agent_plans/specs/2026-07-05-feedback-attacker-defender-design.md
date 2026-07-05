# Attacker/defender primitive design

## Feedback item

Defender counting is a weak primitive behind hanging-piece, capture-quality, and
exchange decisions. The repo already has `3.3_attacked_defended`, but its answer
was loose prose and did not expose counts cleanly.

## Scope

- Keep the existing Tier 3 task.
- Convert answers to fixed five-line grammar with explicit counts.
- Add query-square identity so multiple square probes from one FEN are distinct.
- Store attacker/defender counts and square lists in metadata.

## Answer grammar

```text
Square: <square>
Occupant: <empty|color piece>
White attackers (<n>): <pieces|none>
Black attackers (<n>): <pieces|none>
Defenders (<n>): <pieces|none|not applicable>
```

## Non-goals

- Do not change benchmark composition in this slice.
- Do not add static exchange evaluation yet.
