#!/usr/bin/env python3
"""Synchronize independent RoboTwin task-shard managers across nodes."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

PROGRESS_INTERVAL_SECONDS = 60
CONNECT_RETRY_SECONDS = 1
MAX_MESSAGE_BYTES = 4096


def _recv_line(connection: socket.socket, deadline: float) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out while receiving synchronization message")
        connection.settimeout(min(remaining, PROGRESS_INTERVAL_SECONDS))
        chunk = connection.recv(min(1024, MAX_MESSAGE_BYTES - size))
        if not chunk:
            raise ConnectionError("peer closed before sending a complete message")
        chunks.append(chunk)
        size += len(chunk)
        if size >= MAX_MESSAGE_BYTES:
            raise ValueError("synchronization message is too large")
        payload = b"".join(chunks)
        if b"\n" in payload:
            return payload.split(b"\n", 1)[0]


def _encode_message(*, phase: str, rank: int, exit_code: int) -> bytes:
    return (
        json.dumps(
            {"phase": phase, "rank": rank, "exit_code": exit_code},
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _run_coordinator(args: argparse.Namespace, deadline: float) -> int:
    statuses = {args.rank: args.local_exit_code}
    connections: dict[int, socket.socket] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", args.master_port))
        server.listen(args.world_size - 1)
        next_progress = time.monotonic() + PROGRESS_INTERVAL_SECONDS
        while len(statuses) < args.world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(set(range(args.world_size)) - statuses.keys())
                raise TimeoutError(
                    f"timed out waiting for ranks {missing} during {args.phase}"
                )
            server.settimeout(min(remaining, 5))
            try:
                connection, _ = server.accept()
            except TimeoutError:
                if time.monotonic() >= next_progress:
                    print(
                        f"[h100-multinode] waiting phase={args.phase} "
                        f"ranks={sorted(statuses)}/{args.world_size}",
                        flush=True,
                    )
                    next_progress = time.monotonic() + PROGRESS_INTERVAL_SECONDS
                continue

            try:
                payload = json.loads(_recv_line(connection, deadline))
                peer_phase = str(payload["phase"])
                peer_rank = int(payload["rank"])
                peer_exit_code = int(payload["exit_code"])
                if peer_phase != args.phase:
                    raise ValueError(
                        f"phase mismatch: expected {args.phase}, got {peer_phase}"
                    )
                if not 0 <= peer_rank < args.world_size or peer_rank == args.rank:
                    raise ValueError(f"invalid peer rank: {peer_rank}")
                if peer_rank in statuses:
                    raise ValueError(f"duplicate peer rank: {peer_rank}")
                if peer_exit_code < 0:
                    raise ValueError(f"invalid peer exit code: {peer_exit_code}")
                statuses[peer_rank] = peer_exit_code
                connections[peer_rank] = connection
            except Exception:
                connection.close()
                raise

        global_exit_code = max(statuses.values())
        response = _encode_message(
            phase=args.phase,
            rank=args.rank,
            exit_code=global_exit_code,
        )
        for connection in connections.values():
            with connection:
                connection.sendall(response)
        return global_exit_code


def _run_worker(args: argparse.Namespace, deadline: float) -> int:
    next_progress = time.monotonic() + PROGRESS_INTERVAL_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out connecting to rank 0 during {args.phase}"
            )
        try:
            connection = socket.create_connection(
                (args.master_addr, args.master_port),
                timeout=min(remaining, 5),
            )
            break
        except OSError:
            if time.monotonic() >= next_progress:
                print(
                    f"[h100-multinode] waiting phase={args.phase} "
                    f"for rank 0 at {args.master_addr}:{args.master_port}",
                    flush=True,
                )
                next_progress = time.monotonic() + PROGRESS_INTERVAL_SECONDS
            time.sleep(min(CONNECT_RETRY_SECONDS, max(deadline - time.monotonic(), 0)))

    with connection:
        connection.sendall(
            _encode_message(
                phase=args.phase,
                rank=args.rank,
                exit_code=args.local_exit_code,
            )
        )
        response = json.loads(_recv_line(connection, deadline))
    if str(response["phase"]) != args.phase:
        raise ValueError(
            f"phase mismatch: expected {args.phase}, got {response['phase']}"
        )
    return int(response["exit_code"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--local-exit-code", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    args = parser.parse_args()

    if args.world_size < 2:
        raise ValueError("--world-size must be at least 2")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world-size")
    if not 1 <= args.master_port <= 65535:
        raise ValueError("--master-port must be between 1 and 65535")
    if args.local_exit_code < 0:
        raise ValueError("--local-exit-code must be non-negative")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")

    deadline = time.monotonic() + args.timeout_seconds
    if args.rank == 0:
        global_exit_code = _run_coordinator(args, deadline)
    else:
        global_exit_code = _run_worker(args, deadline)

    print(
        f"[h100-multinode] synchronized phase={args.phase} "
        f"rank={args.rank}/{args.world_size} global_exit_code={global_exit_code}",
        flush=True,
    )
    if global_exit_code != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
