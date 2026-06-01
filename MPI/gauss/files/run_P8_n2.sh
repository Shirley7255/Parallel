#!/bin/sh
#PBS -N G_P8_n2
#PBS -e G_P8_n2.e
#PBS -o G_P8_n2.o
#PBS -l nodes=2:ppn=4

NODES=$(cat $PBS_NODEFILE | sort | uniq)
for node in $NODES; do
scp master_ubss1:/home/${USER}/gauss/main ${node}:/home/${USER} 1>&2
done
/usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/${USER}/main
