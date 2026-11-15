"""Batched simulator IPC: JSON compatibility or framed float32 observations."""

import json
import struct
import subprocess
from pathlib import Path

import numpy as np

from .model import ACTION_SIZE, ENCODER_VERSION, STATE_SIZE

PROTOCOL = "retruco-selfplay-binary-v1"
MAX_REQUEST = 4 << 20
MAX_RESPONSE = 64 << 20


def decode_binary(data):
    if len(data) < 8 or data[:4] != b"RTO1":
        raise RuntimeError("unsupported simulator response format")
    (metadata_size,) = struct.unpack_from("<I", data, 4)
    offset = 8 + metadata_size
    if offset > len(data):
        raise RuntimeError("truncated simulator metadata")
    results = json.loads(data[8:offset])
    if not isinstance(results, list) or not 1 <= len(results) <= 64:
        raise RuntimeError("invalid simulator response batch")
    for result in results:
        observation = result.get("observation")
        if observation is None:
            if not result.get("error"):
                raise RuntimeError("missing simulator observation")
            continue
        count = observation.pop("action_count")
        if (
            type(count) is not int
            or not 0 <= count <= 1024
            or observation["version"] != ENCODER_VERSION
        ):
            raise RuntimeError("incompatible simulator observation")
        size = STATE_SIZE + count * ACTION_SIZE
        if offset + size * 4 > len(data):
            raise RuntimeError("truncated simulator numeric data")
        # Own only this observation's numeric storage. Retaining a learner
        # trajectory must not retain metadata or all other seats in this frame.
        numeric = np.frombuffer(data, dtype="<f4", count=size, offset=offset).astype(
            np.float32, copy=True
        )
        if not np.isfinite(numeric).all():
            raise RuntimeError("nonfinite simulator features")
        observation["features"] = numeric[:STATE_SIZE]
        observation["actions"] = numeric[STATE_SIZE:].reshape(count, ACTION_SIZE)
        offset += size * 4
        training = observation.get("training")
        if training is not None:
            count = training.pop("event_count")
            if (
                type(count) is not int
                or not 0 <= count <= 32
                or training.get("version") != 1
            ):
                raise RuntimeError("incompatible training memory")
            size = count * STATE_SIZE + 128
            if offset + size * 4 > len(data):
                raise RuntimeError("truncated training data")
            numeric = np.frombuffer(
                data, dtype="<f4", count=size, offset=offset
            ).astype(np.float32, copy=True)
            if not np.isfinite(numeric).all():
                raise RuntimeError("nonfinite training data")
            training["memory"]["events"] = numeric[: count * STATE_SIZE].reshape(
                count, STATE_SIZE
            )
            training["privileged"] = numeric[count * STATE_SIZE :]
            offset += size * 4
    if offset != len(data):
        raise RuntimeError("unexpected trailing simulator data")
    return results


class Simulator:
    def __init__(self, executable=None, transport="json"):
        if transport not in ("json", "binary"):
            raise ValueError("unknown simulator transport")
        executable = executable or Path(__file__).resolve().parents[2] / "bin/selfplay"
        self.transport = transport
        self.process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        if transport == "binary":
            try:
                self.process.stdin.write(
                    json.dumps([{"op": "protocol", "protocol": PROTOCOL}]).encode()
                    + b"\n"
                )
                self.process.stdin.flush()
                ack = json.loads(self.process.stdout.readline(MAX_REQUEST + 1))
                if ack != {
                    "protocol": PROTOCOL,
                    "state_size": STATE_SIZE,
                    "action_size": ACTION_SIZE,
                }:
                    raise RuntimeError(
                        "simulator lacks binary IPC; rebuild bin/selfplay or use --ipc json"
                    )
            except BaseException:
                self.close()
                raise

    def _read_exact(self, size):
        parts = bytearray()
        while len(parts) < size:
            part = self.process.stdout.read(size - len(parts))
            if not part:
                raise RuntimeError("simulator response ended before the complete frame")
            parts.extend(part)
        return parts

    def call(self, requests):
        data = json.dumps(requests, separators=(",", ":")).encode()
        if (
            not data
            or len(data) > MAX_REQUEST
            or (self.transport == "binary" and not 1 <= len(requests) <= 64)
        ):
            raise ValueError("simulator request exceeds protocol bounds")
        self.process.stdin.write(
            (struct.pack("<I", len(data)) + data)
            if self.transport == "binary"
            else data + b"\n"
        )
        self.process.stdin.flush()
        if self.transport == "binary":
            (size,) = struct.unpack("<I", self._read_exact(4))
            if not 8 <= size <= MAX_RESPONSE:
                raise RuntimeError("invalid simulator response length")
            results = decode_binary(self._read_exact(size))
        else:
            line = self.process.stdout.readline(MAX_RESPONSE + 1)
            if not line:
                raise RuntimeError(f"simulator exited: {self.process.poll()}")
            if len(line) > MAX_RESPONSE:
                raise RuntimeError("simulator JSON response too large")
            results = json.loads(line)
        for result in results:
            if result.get("error"):
                raise RuntimeError(result["error"])
        if [r["id"] for r in results] != [r.get("id", 0) for r in requests]:
            raise RuntimeError("simulator response ordering mismatch")
        return [result["observation"] for result in results]

    def close(self):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        finally:
            self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
