# Verification Record

Date: 2026-10-10, after splitting the human and mouse research repositories.

- 279 HPO tests passed after aligning the support-first policy.
- Ruff checks passed.
- HPO-only input rejects MP IDs and nonempty mouse-model context.
- Animal-study evidence and human-target inference remain supported.
- Public schemas exported: rich storage 2.0, scientific submissions compact_v2.
- CLI help was exercised under the new hpo_agent package and hpo-agent entry point.
- The previous HP:0009511 compact_v2 result can still be loaded read-only. It retains
  its recorded schema version 1.8, completed status and five observations.
- Historical runs and archives were not rewritten.
- Complete rich-record and compact Root/eval/task workflows finish with supporting
  searches alone. Optional unfetched challenge sources do not block evaluation;
  missing supporting sources still stop before a paid model call.
- Negative evidence labels do not override a defensible review. Explicit scientific
  blockers and misinterpreted studies still prevent release.
- Search-budget validation reserves discovery plus support, without a challenge quota.

Four mouse-specific test cases were removed with their runtime feature; the retained
suite continues to exercise the complete human Root/eval/task research workflow.

Pre-split backup: archives/pre_repository_split_20261010.tar.gz

SHA-256: ab062c912b3329135861a84a0ca43059d4496b2501929f4a1cc6ff5c3d264fc4
