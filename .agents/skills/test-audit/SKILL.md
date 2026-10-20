---
name: test-audit
description: Use when writing, editing, reviewing, or auditing tests in this project. Apply a behavior-focused authoring gate and identify low-value, implementation-coupled, or duplicate coverage and unnecessary test-only production seams.
---

# Test Audit

Optimize for confidence in behavior, not test count or deletion count. Use the
same value bar in three modes:

- **Authoring:** apply the gate below to every new or changed test. Keep the
  work scoped to the requested change; ordinary test edits do not start an audit.
- **Audit:** inspect a focused set of tests and simplify coverage with evidence.
- **Campaign:** audit a subsystem's complete test surface. Read
  [CAMPAIGN.md](CAMPAIGN.md) before starting this mode.

## Authoring gate

Before adding or changing a test, answer these questions:

1. What observable behavior, invariant, or independent contract does it protect?
2. What credible regression makes it fail?
3. Why does existing coverage not already catch that failure? Give each contract
   a primary test owner at the boundary that proves it best. Another layer must
   cover a distinct risk, such as transport, persistence, or lifecycle behavior
   the primary test cannot reach. Extend a parameterized case or shared fixture
   when that is clearer than adding a near-duplicate test.
4. Does the test require a production export, flag, wrapper, or injection hook
   used only by tests? Prefer the real boundary or an existing production
   abstraction; do not introduce a seam solely to expose implementation details.

A missing answer means the test design needs more work. Check the design
against the low-value patterns below. A match needs an independently meaningful
contract under the retention bar; otherwise rewrite at the owning boundary.
Behavior-preserving refactors should not break behavior tests.

For bug regressions, demonstrate that the test fails on the pre-fix behavior for
the intended reason and passes with the fix. Use an isolated checkout or a
carefully restored local control without disturbing unrelated work. If the
control cannot run, report that limitation rather than claiming verified
regression protection. Do not replay the same bug at every layer unless each
test protects a distinct failure mode.

## Low-value patterns

These are investigation signals, not automatic reasons to delete a test:

- Assertion-free coverage probes, self-comparisons, and identity copiers.
- Copied fixtures, inventories, manifests, or export lists that merely restate
  the implementation.
- Exact source, import, or string checks without an independent contract.
- Private predicate or call-shape assertions duplicated at a real boundary.
- Repeated invocations of the same contract, including module-local replays of
  shared helpers.
- Tests that only preserve test-only exports, globals, wrappers, or otherwise
  dead production paths.
- Expected values computed by the helper or renderer under test.
- Mocks that implement the behavior being asserted, or identical mocks used
  for collaborators with different contracts.
- Fixtures that supply the state transition, ordering, or persistence the
  production owner should produce; assertions against a store the path never
  writes.
- Capability checks that restate declared flags instead of exercising the
  behavior those flags promise.
- Negative controls that pass for an unrelated reason, such as rejection by a
  different guard or a path the production code never reaches.
- Names and fixtures that promise more than the assertions actually exercise.

## Value and retention bar

For JSON v2 fixture decoders, map the actual member names with explicit `json`
tags. Require the independently captured, nonzero case count before accepting an
oracle result; an empty decoded corpus must fail. A loop that executes zero
cases provides no parity evidence even if the command exits successfully.

Keep tests that independently protect behavior, a credible regression, or a
public API, protocol, configuration, migration, storage, security, platform,
default, generated artifact, packaging, release, or architecture contract.
Observable call ordering and exact externally meaningful bytes can be contracts.

Static or slow is not a deletion reason. Source inspection can be the cheapest
independent guard when it checks a real contract and survives an identifier-only
refactor. An existing test that breaks during behavior-preserving reorganization
is suspect, not automatically disposable.

Before judging an audit candidate, read the complete test, production owner,
entry points, callers, callees, overlapping tests, relevant sibling
implementations, CI routing, and history. Read applicable root and scoped
`AGENTS.md` files first. For dependency-backed claims, inspect the installed
dependency source, types, or authoritative documentation as needed.

A retained test that fails at baseline may expose a product defect or an
environment problem. Reproduce and classify it; do not delete it to make the
suite pass. Repair product defects only within the authorized scope.

## Discovery and candidate evidence

Keep discovery read-only and record evidence before editing. For focused audits,
prefer a few high-confidence candidates over a speculative inventory. Organize
broad discovery by production ownership, including shared boundaries and test
support, rather than assuming directory names define ownership. Delegation is
optional and subject to the current task's authorization.

For each proposed removal or consolidation, record:

- Exact test name and location, and the failure it can actually detect.
- Non-test callers of any production or support seam proposed for removal.
- Stronger remaining boundary coverage, or why no contract needs coverage.
- Relevant history and why the test or seam exists; note unavailable history.
- Production or support cleanup unlocked, if any.
- Risk and the focused validation command.

Unresolved evidence means the candidate is not ready for deletion.

## Edit shape

Choose one coherent batch around a production owner. Move retained assertions
into their keeper tests before removing redundant coverage. Consolidate repeated
setup when it improves clarity, without creating a new opaque harness.

Remove obsolete test-only seams and dead production paths only after checking
their consumers and contracts. Reduced production complexity is useful, but
net-negative line counts are not a quota. Do not replace deleted tests with new
tests that restate the same implementation or broaden a focused edit into cleanup.

## Validation in this project

Discover current commands and prerequisites from repository documentation,
package configuration, and CI rather than assuming a single test runner.

Do not edit source or tests while a validation run is reading that checkout.
Use the documented disposable test environment for integration tests. For
removed source greps or plan assertions, exercise the executable behavior or
dry-run that owns the contract. Run relevant formatting or lint checks and
`git diff --check`, then inspect the final diff for unintended coverage loss.

For audits, inspect `git diff --numstat` and distinguish production/tooling from
tests and test support. Report the commands actually run, results, skips, and
limitations. Perform any review required by repository policy; this skill does
not depend on a particular external review tool.

## Completion and continuation

Commit, push, open a PR, or merge only when authorized and using the repository's
workflow. A focused change can finish locally. Continue a broad audit in coherent
batches only within the requested scope, refreshing discovery after relevant
upstream changes.

For ordinary test edits, report the protected behavior and validation. For
audits, also report removed low-value categories, production simplifications,
retained false positives, remaining coverage, production versus test/support
line counts, PR state if applicable, and named follow-ups. Distinguish verified
results from intended coverage.
