#!/usr/bin/env python3

import os
import re
import glob
from collections import defaultdict

# 匹配格式：[RESULT] ALG: Serial, N: 500, P: 1, EXEC_TIME: 0.123456
PATTERN = re.compile(
    r"\[RESULT\]\s+ALG:\s+(\S+),\s+N:\s+(\d+),\s+P:\s+(\d+),\s+EXEC_TIME:\s+([\d.e+\-]+)"
)

# 算法显示顺序
ALG_ORDER = ["Serial", "MPI_Block", "MPI_Cyclic", "MPI_Hybrid"]


def main():
    # 收集所有 .o 文件中的 [RESULT] 行
    records = []
    for fname in sorted(glob.glob("*.o")):
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if line.startswith("[RESULT]"):
                    m = PATTERN.match(line)
                    if m:
                        records.append({
                            "alg": m.group(1),
                            "N": int(m.group(2)),
                            "P": int(m.group(3)),
                            "time": float(m.group(4)),
                        })

    if not records:
        print("未找到任何 [RESULT] 行。")
        return

 
    groups = defaultdict(list)
    for r in records:
        groups[r["alg"]].append(r)


    for alg in groups:
        groups[alg].sort(key=lambda r: (r["N"], r["P"]))

   
    for alg in ALG_ORDER:
        if alg not in groups:
            continue
        items = groups[alg]
     
        for i, r in enumerate(items):
            alg_display = alg if i == 0 else ""
            print(f"| {alg_display:<16} | {r['N']:<14} | {r['P']:<10} | {r['time']:<15.6f} |")
   
        print()


    for alg in groups:
        if alg in ALG_ORDER:
            continue
        items = groups[alg]
        for i, r in enumerate(items):
            alg_display = alg if i == 0 else ""
            print(f"| {alg_display:<16} | {r['N']:<14} | {r['P']:<10} | {r['time']:<15.6f} |")
        print()


if __name__ == "__main__":
    main()
