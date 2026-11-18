#!/usr/bin/env python3
"""Per-GPU table: utilisation, used/free VRAM and the users (with their memory and process count) on each GPU."""
import subprocess
from collections import defaultdict


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()


def main():
    gpus = []
    for line in run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"]):
        idx, uuid, name, tot, used, util = [x.strip() for x in line.split(",")]
        gpus.append((int(idx), uuid, name, int(tot), int(used), int(util)))
    users = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # uuid -> user -> [MiB, n procs]
    for line in run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]):
        uuid, pid, mem = [x.strip() for x in line.split(",")]
        user = (run(["ps", "-o", "user=", "-p", pid]) or ["?"])[0].strip()
        users[uuid][user][0] += int(mem)
        users[uuid][user][1] += 1
    rows = []  # each entry: list of text lines per cell
    for idx, uuid, name, tot, used, util in gpus:
        u = [f"{k} {v[0] / 1024:.1f} GiB ({v[1]})" for k, v in sorted(users[uuid].items(), key=lambda kv: -kv[1][0])] or ["-"]
        rows.append([[str(idx)], [f"{util}%"], [f"{used / 1024:.1f}"], [f"{(tot - used) / 1024:.1f}"], u])
    head = [["GPU"], ["Util"], ["Used GiB"], ["Free GiB"], ["Users (VRAM, #procs)"]]
    w = [max(len(l) for r in rows + [head] for l in r[i]) for i in range(len(head))]
    last = len(head) - 1
    cell = lambda c, i: f" {c.ljust(w[i]) if i == last else c.rjust(w[i])} "
    line = lambda l, m, r: l + m.join("\u2500" * (x + 2) for x in w) + r

    def emit(r):
        for k in range(max(len(c) for c in r)):
            print("\u2502" + "\u2502".join(cell(c[k] if k < len(c) else "", i) for i, c in enumerate(r)) + "\u2502")

    print(line("\u250c", "\u252c", "\u2510"))
    emit(head)
    for r in rows:
        print(line("\u251c", "\u253c", "\u2524"))
        emit(r)
    print(line("\u2514", "\u2534", "\u2518"))


if __name__ == "__main__":
    main()
