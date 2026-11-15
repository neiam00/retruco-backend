# Training the Retruco CPU

This Python project trains policies against the same Go rules engine used by the game server. The simulator supports heads-up and team games, including cantos, flor, and optional team signals. Each actor chooses among the legal moves available to its seat. Recurrent training also uses a centralized critic, which stays on the training machine.

## Run a first experiment

Use a backend checkout on Linux or macOS with Go, uv, and Python 3.13. From the repository root:

```sh
mkdir -p bin
go build -o bin/selfplay ./cmd/selfplay
cd ml
uv sync --locked --python 3.13
uv run --locked retruco-train --device cpu --workers 1 --environments 2 \
  --players 4 --steps 128 --updates 1 --output runs/quickstart
uv run --locked python -m retruco_ml.evaluate runs/quickstart/checkpoint.pt \
  --seed 100000 --pairs 2 --output runs/quickstart/evaluation.json
```

This runs one small population PPO update from scratch and exercises paired evaluation on development deals. The report records seat rotations, match targets, flor settings, and win rates. Use a fresh output directory for a different experiment. To continue the baseline, repeat its training command with `--resume`. The `--updates` option counts additional updates for that invocation.

## Recurrent and team training

`retruco_ml.rmappo_train` trains a GRU64 or LSTM64 actor with separate hand memory for each seat. The actor receives its own hand, public plays and cantos, and signals delivered to that seat. The critic can use additional information during training. Learners on the same team receive the team result and keep separate decision histories.

The recurrent trainer requires a frozen input pool. Supply a `manifest.json` with a `reference` entry and `parents` entries for seeds 101, 102 and 103. Each entry specifies a checkpoint filename relative to the manifest and its SHA-256. The reference is a feed-forward policy and the three parents are recurrent opponents. Seed 103 is reserved for evaluation. Historical checkpoints and generated reports are local experiment inputs. The feed-forward quickstart above works with a fresh checkout.

For example, with the pool under `ml/runs/inputs/`, run this from `ml/`:

```sh
uv run --locked python -m retruco_ml.rmappo_train --cell lstm --seed 1101 \
  --simulator ../bin/selfplay --inputs runs/inputs/manifest.json \
  --choices 250000 --output runs/lstm-1101
```

Training budgets count genuine choices. A collection batch finishes its matches and can exceed the requested count. The status and metrics files record the actual count. Use the trainer's `--resume` option with the same inputs and configuration to restore a saved run.

The [experiment configurations](experiments) specify opponent mixtures, checkpoint intervals, deal ranges and comparison criteria. Continuation recipes identify the exact parents and sometimes a prior audit report by hash. Paths beginning with `ml/` refer to the backend checkout. Supply those inputs to reproduce a historical recipe, or create a new configuration for your own checkpoints. Generated runs belong in `runs/` and local audit reports in `reports/`. The [simulator protocol](IPC.md) describes the observation and legal-action transport.

## Evaluate and export a candidate

Compare policies on the same deals across learner seats, targets of 12 and 24, and flor settings. The native benchmark measures heads-up wins separately from the equal-weight average over four team partners: literal, trusting, cautious and reference-human. These partners are programs. The [benchmark definition](experiments/strength-targets-v2.json) gives the thresholds and confidence-interval criteria. Keep selection deals separate from final evaluation and record the rules revision with the result.

The runtime is a Rust library compiled for native execution and WebAssembly. Install Rust and Node to run the parity checks. Build from the runtime directory so Cargo picks up the WASM memory limit:

```sh
rustup target add wasm32v1-none
(cd runtime && cargo build --locked --release && cargo build --locked --release --target wasm32v1-none)
```

From `ml/`, export a supported recurrent checkpoint with the same choice policy used for evaluation. This example uses the current team policy. Set `CHECKPOINT` to the checkpoint file and `TRAINING_SNAPSHOT` to the matching frozen source manifest.

```sh
uv run --locked python -m retruco_ml.bundle "$CHECKPOINT" runs/browser-candidate \
  --training-snapshot "$TRAINING_SNAPSHOT" --team-play closing-card-and-flor-v2 \
  --choice-policy value-and-team-v9
```

The bundle contains a quantized actor, the WASM evaluator, and a manifest with their hashes and source identity. Use the frontend's [local CPU preview instructions](https://github.com/neiam00/retruco-webapp#local-cpu-evaluation) to play it. Production loading requires an approved bundle. Keep the corresponding simulator, training source revision, lockfile and checkpoint together for later verification.

## Development checks

From `ml/`, with the simulator and Rust targets built:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
```

Run `uv run --locked ruff check --fix .` and `uv run --locked ruff format .` to apply safe lint fixes and formatting. Ruff excludes frozen runs and generated build outputs.
