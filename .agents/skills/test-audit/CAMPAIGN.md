# Subsystem test-audit campaign

Use this workflow when the requested scope covers a subsystem's whole test
surface. A subsystem may be an API domain, frontend feature, shared library,
or operational workflow. The value bar, retention bar, candidate evidence, and
validation in [SKILL.md](SKILL.md) apply throughout. Organize the result as one
coherent change or a sequence of reviewable changes according to repository
policy and the user's scope.

Each stage ends with its completion criterion. Do not begin destructive cutover
before the inventory and preservation plan are complete.

## 1. Establish a baseline

Record the baseline commit, relevant working-tree changes, subsystem test and
support line counts, and every in-scope test file's result. Record passing,
failing, skipped, and blocked cases separately, including environment and
dependency prerequisites. A pre-existing failure is evidence to investigate,
not a reason to discard a test.

Done when every in-scope test file has a recorded baseline status and any
missing execution is explicitly accounted for.

## 2. Inventory ownership

Partition the surface by production responsibility, not file prefixes. Include
cases at shared boundaries, integration and browser scenarios, operational
checks, and supporting harnesses. Assign each test to one primary lane and
record cross-lane dependencies separately.

Done when every in-scope file and scenario has an owner, with shared support
identified so it cannot be changed inconsistently.

## 3. Build a read-only ledger

Read every assigned test, including parameter tables, together with its
production owners, entry points, callers, history, and CI routing. Record each
test declaration with one mark. A parameterized test can be one entry unless
its rows need different decisions.

- `R`: retain; name the contract and credible regression. Note moves separately.
- `F`: retain the contract but repair a weak, vacuous, or misdirected assertion.
- `C`: consolidate; name the keeper that will absorb its distinct assertions.
- `D`: delete; name the coverage that remains, or explain why no contract exists.

Judge assertions rather than names. Include the candidate evidence required by
`SKILL.md`; leave uncertain candidates retained pending investigation. Lanes may
be inspected sequentially or by separate read-only agents when delegation is
authorized and available.

Done when every declaration has a mark and evidence, with uncertainty explicit.

## 4. Plan which layers remain

Use a second read-only pass to examine redundant layers across the ledger. Name
a keeper suite for each contract. Prefer exercising the real owner with
controlled external dependencies over mocking away the behavior under test.
Retain tests at multiple layers when they cover distinct failure modes.

Done when every lane names retired files, keepers, assertions to transfer,
test-only seams eligible for removal, and any corrected ledger decisions.

## 5. Apply the cutover

Edit lane by lane. Coordinate shared harness changes through one owner when
working in parallel. Transfer required assertions before deleting redundant
suites. Remove unlocked test-only seams after checking their consumers.

Update CI routing and inventories if the repository uses explicit registration.
Update coverage or size baselines only when their meaning is preserved; do not
weaken gates to hide lost coverage. Add durable ownership guidance to scoped
`AGENTS.md` files only when concrete findings justify it and the task permits it.

Done when each lane plan is applied and its keepers pass, or documented baseline
failures and environment blockers remain clearly separated from new failures.

## 6. Review preservation

Compare removed coverage against keepers in a separate review pass. Use an
independent reviewer when available and authorized; otherwise record that the
review was performed by the same author. Check for contracts that lost their
only proof and assertions that cannot fail for the intended reason.

For restored or materially repaired coverage, make a deliberate temporary
mutation to the production owner, or run an equivalent pre-fix control, and
confirm the keeper fails for the intended reason. Restore the source exactly
and verify restoration before continuing; isolate controls from unrelated work.

Done when every reported gap is restored or rejected with evidence, and each
restored contract has a failing control. If a control cannot run, report the
unverified contract rather than claiming complete preservation.

## 7. Triage product defects

A baseline failure that survives into a keeper needs diagnosis. Distinguish a
product defect from setup problems or an incorrect test expectation. Fix an
in-scope defect at its production owner as a distinct change, proving a failing
control and a passing candidate through the same harness. Record unrelated
defects as follow-ups instead of expanding the campaign automatically.

Done when defects have evidence and an explicit disposition; every claimed fix
has control and candidate results.

## 8. Reconcile and hand off

If the base branch changes, reconcile using the repository's integration policy.
For upstream changes to a retired test file, inspect the new contract and port
needed proof into its keeper. Reconsider the deletion when its rationale no
longer holds; never resolve conflicts by discarding new coverage blindly.

Rerun affected keepers and the full subsystem suite on the final reconciled
state, including relevant integration proof. Inspect large diffs in bounded
groups so truncated review output does not hide files.

Hand off with the `SKILL.md` audit report plus baseline/final test and support
line counts, production counts separately, lanes and keepers, retired layers,
preservation gaps and controls, defect dispositions, and any blocked or unrun
validation. Do not claim a fully validated campaign while required proof is
missing.
