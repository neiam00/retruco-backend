# Local simulator transport

The Go simulator and Python training tools exchange a 768-value state vector and 48 values per legal action. The PPO trainer uses binary transport by default. Pass `--ipc json` for tools using the JSON protocol.

The client first sends one JSON line:

```json
[{"op":"protocol","protocol":"retruco-selfplay-binary-v1"}]
```

The server answers with a JSON line containing `protocol`, `state_size: 768`, and
`action_size: 48`. This handshake must be the first request. An older executable
returns an immediate error; the client instructs the operator to rebuild instead
of hanging or silently changing transport.

After acknowledgment, each request is a little-endian uint32 byte length followed
by UTF-8 JSON commands. Small control messages remain JSON. Requests are limited
to 4 MiB and 1–64 commands. Responses have a little-endian uint32 length followed
by this payload (at most 64 MiB):

1. Four ASCII bytes `RTO1`.
2. Little-endian uint32 metadata byte length.
3. UTF-8 JSON response metadata in request order. Each observation omits
   `features`/`actions` and includes `action_count` (0–1024).
4. For each successful observation, 768 float32 state values followed by
   `action_count × 48` float32 candidate values, all little-endian and in the
   existing shared-Go encoder order. Error responses have no numeric block.

Terminal observations retain the fixed state block and zero action rows. Python
represents numeric blocks as NumPy float32 arrays, including an empty `(0,48)`
action matrix. Other metadata is unchanged. No legal candidate is truncated.
Response IDs, frame sizes, versions, numeric lengths and finite values are
checked. Truncated frames and trailing data fail instead of being reused.

Each decoded observation owns only its numeric storage. A retained learner
trajectory does not retain the entire IPC frame or other players' metadata.
Packing keeps ragged candidates for the rollout and allocates padding only for
the current minibatch. A batch owns its buffers through autograd; a later forward
pass cannot overwrite them. PPO reuses packed observations across its epochs.

Errors do not trigger automatic action retries. As with the legacy transport,
commands in a batch execute sequentially, so callers must not assume a failed
batch was atomic. Training resumes at a saved complete-update checkpoint.
