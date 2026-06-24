#!/usr/bin/env python3
"""
Detached trainer launcher for rank0.
Usage: python scripts/launch_detached.py <log_file> <wrapper_script>
"""
import sys
import os
import subprocess

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <log_file> <wrapper_script>")
        sys.exit(1)

    log_file = sys.argv[1]
    wrapper_script = sys.argv[2]

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            ['bash', wrapper_script],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    print(proc.pid)

if __name__ == '__main__':
    main()
