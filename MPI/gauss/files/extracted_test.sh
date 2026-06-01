#!/bin/bash
# Usage: bash test.sh [ID] [NODES]

if [ $# -lt 2 ]; then
    echo "参数缺失"
    exit 1
fi

ID=$USER
LAB="${1}"   # 这里如果第一个参数为空，那么LAB为空，需要根据实际情况调整       
NODES="${2}" # 第二个参数表示申请节点数
LOG_PATH="/parallel_hw/gauss/${LAB}/"
RESULT_PATH="/home/${ID}/gauss/"

if [ "$NODES" -gt 12 ]; then
    echo "计算节点申请过多"
    exit 1
fi

rm -f "${RESULT_PATH}test.o" "${RESULT_PATH}test.e"

jobid=$(qsub -l nodes="${NODES}" qsub.sh)
echo "Submitted job with ID: $jobid"

timeout=12000
elapsed=0
interval=5

while [ ! -f "${RESULT_PATH}test.o" ] && [ $elapsed -lt $timeout ]; do        
    sleep $interval
    elapsed=$((elapsed + interval))
done

if [ ! -f "${RESULT_PATH}test.o" ]; then
    echo "Timeout reached, test.o not generated."
    exit 1
fi

# 读取 qsub 脚本的输出文件 /home/${USER}/ann/test.o
if [ -f "${RESULT_PATH}test.o" ]; then
    output_o=$(cat "${RESULT_PATH}test.o")
    output_e=$(cat "${RESULT_PATH}test.e")
else
    output_o="Output file ${RESULT_PATH}test.o not found."
    output_e="Error file ${RESULT_PATH}test.e not found"
fi

#output=$(./main)
echo "${output_e}"

echo "${output_o}"

truncated_output=$(echo "$output_o" | tail -n 10)

current_time=$(date +"%Y-%m-%d-%H-%M-%S")

log_file="${LOG_PATH}${ID}_${LAB}.log"
{
    echo "test time: $current_time"
    echo "$truncated_output"
    echo "----------------------------------------"
    echo ""
} >> "$log_file"
