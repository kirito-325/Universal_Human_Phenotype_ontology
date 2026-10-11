# Training-free HPO-MPO semantic/CPO benchmark

> Exploratory downstream validation, separate from the deprecated structural-embedding method. This code-only snapshot excludes frozen data and result artifacts; the development outcome below describes the original run, not a new evaluation of the current agents.

This directory implements the registered methods in `configs/registry.yaml`
against the independently
frozen 50-disease `mission3` benchmark. It never trains an encoder or fusion
parameter and does not use the gold label inside a term-pair provider.

The execution order is:

```bash
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo audit-resources
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo reproduce-baselines
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo freeze-inputs
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo encode
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo run-stage --stage nodes
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo run-stage --stage atoms
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo run-stage --stage profiles
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo run-stage --stage fusion
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo run-controls
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo report
PYTHONPATH=MPO_HPO_mapping_v4/src python3 -B -m semantic_cpo freeze-method
```

`external-test` refuses to run without a separately supplied cohort and never
invokes development method selection.

## Current development outcome

- Recommended default: pure frozen semantic surface `P0`.
- Exploratory candidate: `node_exact__atom_A2__profile_bma__linear`.
- O0-P0 MRR: `+0.000476` (1 win, 48 ties, 1 loss); bootstrap interval crosses zero.
- O1 and P1 have identical ranking metrics.
- Registered profile and degree-preserving null controls do not support a
  mechanism-specific gain, so no CPO-benefit claim is made.

See `results/semantic_cpo_training_free_v1/report/development_report.md` and
`results/semantic_cpo_training_free_v1/frozen_method/frozen_method.json`.

## Offline checks

From this directory, with the shared scientific dependencies and pytest installed:

```bash
PYTHONPATH=src python -m pytest tests/test_math_contracts.py \
  tests/test_real_execution_contracts.py::RealExecutionContractTests::test_provider_accepts_term_ids_only
```

The remaining execution-contract tests require the original real-data artifacts,
which are intentionally absent from this snapshot.
