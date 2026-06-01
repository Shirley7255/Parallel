#!/bin/sh
#PBS -N G_P16
#PBS -e G_P16.e
#PBS -o G_P16.o
#PBS -l nodes=2:ppn=8

NODES=$(cat $PBS_NODEFILE | sort | uniq)
for node in $NODES; do
scp master_ubss1:/home/${USER}/gauss/main ${node}:/home/${USER} 1>&2
done
/usr/local/bin/mpiexec -np 16 -machinefile $PBS_NODEFILE /home/${USER}/main
