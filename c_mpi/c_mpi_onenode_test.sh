#!/bin/bash
#SBATCH --job-name=c_mpi_onenode_test
#SBATCH --output=c_mpi_onenode_results.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1      
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4   
#SBATCH --cpus-per-task=1     
#SBATCH --mem=2G
#SBATCH --partition=tornado

HOME_DIR="$HOME/parallel_sor"
LINSYS="$HOME_DIR/linsys"
OUTS="$HOME_DIR/outs"
C_MPI_DIR="$HOME_DIR/c_mpi"

W=1.5
N=300
E=0.000000000001

module load mpi 

RESULTS_FILE="c_mpi_onenode_results.csv"
echo "nodes;time_ms" > $RESULTS_FILE

for NODES in {1..4}; do
    echo "$NODES nodes"
    
    START_TIME=$(date +%s%3N) 

    mpirun -np $NODES  "$C_MPI_DIR/build/sor"  \
        -c "$LINSYS/${N}.txt" \
        -o "$OUTS/c_mpi_${N}_${NODES}.txt" \
        -n $N -e $E -w $W

    END_TIME=$(date +%s%3N) 
    RUNTIME=$((END_TIME - START_TIME))

    echo "$NODES;$RUNTIME" >> $RESULTS_FILE
    echo "runtime: $RUNTIME ms"
done
