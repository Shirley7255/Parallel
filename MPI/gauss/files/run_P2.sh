#!/bin/sh
#PBS -N G_P2
#PBS -e G_P2.e
#PBS -o G_P2.o
#PBS -l nodes=1:ppn=8

NODES=$(cat $PBS_NODEFILE | sort | uniq)
for node in $NODES; do
scp master_ubss1:/home/${USER}/gauss/main ${node}:/home/${USER} 1>&2
done
/usr/local/bin/mpiexec -np 2 -machinefile $PBS_NODEFILE /home/${USER}/main
