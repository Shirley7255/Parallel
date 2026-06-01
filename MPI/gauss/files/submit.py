#!/usr/bin/env python3


import os
import sys


# 配置参数

P_list = [1, 2, 4, 8, 16]   # 要测试的进程数
PPN = 8                       # 每节点固定 8 核


SCRIPT_TEMPLATE = """#!/bin/sh
#PBS -N G_P{P}
#PBS -e G_P{P}.e
#PBS -o G_P{P}.o
#PBS -l nodes={nodes}:ppn={ppn}

NODES=$(cat $PBS_NODEFILE | sort | uniq)
for node in $NODES; do
scp master_ubss1:/home/${{USER}}/gauss/main ${{node}}:/home/${{USER}} 1>&2
done
/usr/local/bin/mpiexec -np {P} -machinefile $PBS_NODEFILE /home/${{USER}}/main
"""


def main():
    
    main_binary = "/home/{}/gauss/main".format(os.environ["USER"])
    if not os.path.exists(main_binary):
        print(f"[WARN] {main_binary} 不存在，请先编译：mpic++ -O3 -fopenmp main.cc -o main")
        
    for P in P_list:
        
        if P <= PPN:
            nodes = 1
        else:
            nodes = 2   # P=16 时需 2 节点 × 8 核 = 16 核

        
        script_content = SCRIPT_TEMPLATE.format(P=P, nodes=nodes, ppn=PPN)

        
        script_name = f"run_P{P}.sh"
        with open(script_name, "w") as f:
            f.write(script_content)
        os.chmod(script_name, 0o755)  

        
        print(f"[SUBMIT] P={P}, nodes={nodes}, ppn={PPN} → qsub {script_name}")
        ret = os.system(f"qsub {script_name}")
        if ret != 0:
            print(f"[ERROR] qsub 返回码 {ret}，请检查 PBS 环境", file=sys.stderr)

    print("\n所有作业已提交。完成后查看 G_P*.o 文件获取结果。")


if __name__ == "__main__":
    main()
