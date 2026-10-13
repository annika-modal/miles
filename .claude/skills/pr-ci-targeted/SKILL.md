---
name: pr-ci-targeted
description: Pick the GPU CI a miles PR actually needs and drive it to green — list the registered test files the diff can reach, run them one file at a time with `/rerun-test` comments (or a minimal set of `run-ci-*` labels when the change is broad), then fix, push and re-run only the failing files until they pass. Use when asked which CI labels a PR needs, to "run the CI for my change", to babysit a PR's GPU tests, or to iterate a fix against CI.
---

# Targeted PR CI

GPU runners are scarce: a domain label runs every test carrying it, and every
`labeled` event restarts `PR Test`. Most PRs only need the handful of test files
their diff can reach. Run those files directly, read the result on the PR, fix,
and re-run just what failed.

Background: labels and `/rerun-test` are specified in
[docs/developer/ci/01-label.md](../../../docs/developer/ci/01-label.md), test
registration in [ci-test-registration.md](../../rules/ci-test-registration.md),
and failure handling in [ci-failure-triage.md](../../rules/ci-failure-triage.md).

## 1. List the affected tests

From the repo root, on the PR branch:

```bash
PYTHONPATH=. python3 .claude/skills/pr-ci-targeted/affected_tests.py --base origin/main
```

It prints:
- every registered test the diff reaches, with backend, suite, labels, hardware,
  `est_time`, disabled reason, and the changed file it was reached from;
- one ready-to-post `/rerun-test <file>` line per enabled CUDA test and per
  labeled CPU test;
- the domain labels of those tests, as "reached / all tests with that label";
- hubs: files whose dependents reach more than `--hub-limit` (40) enabled CUDA
  tests, such as shared arguments or launcher utils. They are not followed. A
  changed hub means the change is broad, unless the diff there only adds
  something new (a flag read by the files already listed); judge it by what
  changed.
- changed non-Python paths outside the no-GPU allowlist of
  `tests/ci/stage_selection.py` (Docker, workflow, config files): these affect
  GPU runtime outside the import trace and need label-based CI, even when a CPU
  test mentions them.
- changed files that `train.py`, `train_async.py`, `train_multi_policy.py` or
  `serve_tinker.py` load: every e2e training run executes them, but e2e tests
  start those entry points through the launcher hub, so the trace cannot list
  them. Add the domain labels of the training paths the change touches.
- changed Python files nothing imports or names: entry points, or code loaded
  some other way.

The trace builds the import graph with `ast`: absolute, relative and
`from pkg import module` imports, bare imports of a sibling file put on
`sys.path`, and dotted module paths or file paths inside string literals
(`--custom-loss-function-path`, `python tools/x.py` subprocesses). A model
definition under `scripts/models/` is matched by its name in string literals
(`--model-name`), including variants built from it (`-5layer`, `_lora`). Read
the list before acting.
- **Drop false positives**: a test that only names a path in an error message,
  or a long multi-policy run reached through a shared launcher helper.
- **Add misses**: other dynamic loading (`importlib` with a computed name,
  plugin registries), or a test whose model uses the changed kernel through a
  path the trace did not follow. Grep the tests for the model or op name.

CPU tests without labels (`stage-a-cpu`, `stage-b-cpu`) run on every PR; they
need no action. A labeled CPU registration (`register_cpu_ci(labels=[...])`,
e.g. `rollout`, `k8s`, `agentic`) is gated on `run-ci-<label>` like a GPU test,
so it is listed with the `/rerun-test` lines.

## 2. Choose files or labels

- **A few GPU files (roughly ≤ 10), no changed hub** → `/rerun-test`, one
  comment per file. This is the default.
- **Many files from one domain, a changed hub, a changed file on the training
  entry path, or a path outside the import trace** (arguments, launcher utils,
  Docker or workflow files) → the smallest set of domain labels that covers
  them. Prefer a label whose "reached / all" ratio is high. Add `run-on-blackwell`
  only when Blackwell-specific code changed.
- **Don't add labels that are not needed.** Skip `bypass-fastfail` unless you
  need every failure from one run.

Spending runners is outward-facing. Propose the exact comments or labels and
wait for the user's go-ahead, unless the user already asked you to run CI for
this PR.

## 3. Run one file per comment

The comment must be exactly the command; `/rerun-test` takes a single path:

```bash
gh pr comment <N> -R radixark/miles --body "/rerun-test tests/fast-gpu/test_x.py"
```

How the run behaves:
- It runs on the test's home suite, at the PR head SHA at dispatch time.
- Domain labels and the nightly gate do not apply.
- A disabled, unregistered or ROCm-only file fails the resolve job.

The workflow reacts with 👍, then posts a status comment. It edits that comment
to ✅ / ❌ / ⚪ with the elapsed time when the run ends.

To watch all of a PR's file runs at once:

```bash
gh run list -R radixark/miles --workflow "Rerun Test" --limit 40 \
  --json databaseId,status,conclusion,displayTitle \
  -q '.[] | select(.displayTitle | test("PR #?<N>\\b")) | "\(.status) \(.conclusion) \(.displayTitle)"'
```

Poll every 10–30 minutes, depending on the suites' `est_time`. Do not poll in a
tight loop.

## 4. Fix and re-run until green

For each ❌:
1. Get the complete log with the `ci-fetch-log` skill.
2. Classify the failure with [ci-failure-triage.md](../../rules/ci-failure-triage.md),
   using the log line itself.
3. If the PR caused it: fix it, reproduce locally or on a devbox when you can,
   and push. Follow the repo's push policy (append commits; force-push only
   where the PR allows it).
4. Post one short note on the PR (what failed, why, and the fix commit),
   followed by `/rerun-test <file>` for that file only. Files that already
   passed and are untouched by the fix need no re-run.
5. If it is infra or a flake: re-run the file once. If it fails the same way,
   report it as the triage rule says.

Each push makes earlier results stale for files the push touches. Before calling
the PR green, re-run any file whose reached-from source changed after its last
pass.

## 5. Record timing for new tests

A new or moved CUDA test needs a measured `est_time`; see `ci-test-registration.md`.
1. Read **Run tests** from its `/rerun-test` run.
2. Set `est_time = ceil(1.25 × measured)`, rounded up per the rule.
3. Add the PR-description line, for example:
   `CI timing: tests/fast-gpu/test_x.py on stage-b-2-gpu-h200, Run tests 41 s, est_time=60, https://github.com/radixark/miles/actions/runs/<id>`.

## Report

Tell the user:
- which files were run, and each one's result and elapsed time;
- what failed, why, and which commit fixed it;
- what was deliberately not run, and why.
