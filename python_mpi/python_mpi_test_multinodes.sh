#!/bin/bash
#SBATCH --job-name=python_mpi_multinode_test
#SBATCH --output=python_mpi_multinodes_results.log
#SBATCH --time=00:40:00  
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1   
#SBATCH --cpus-per-task=1     
#SBATCH --mem=2G
#SBATCH --partition=tornado

HOME_DIR="$HOME/parallel_sor"
LINSYS="$HOME_DIR/linsys"
OUTS="$HOME_DIR/outs"
PYTHON_MPI_DIR="$HOME_DIR/python_mpi"

W=1.5
N=2000
E=0.000000000001

module load python/3.12 

RESULTS_FILE="python_mpi_multinodes_results.csv"
echo "nodes;time_ms" > $RESULTS_FILE


REPEATS=3 

for INSTANCES in {1..4}; do
    echo "$INSTANCES nodes"
    TOTAL_TIME=0
    
    for ((i=1; i<=REPEATS; i++)); do
        START_TIME=$(date +%s%3N) 

        mpirun -np $INSTANCES  python3.12 "$PYTHON_MPI_DIR/python_mpi.py"  \
        -c "$LINSYS/${N}.txt" \
        -n $N -e $E -w $W
        
        END_TIME=$(date +%s%3N) 
        RUNTIME=$((END_TIME - START_TIME))
        TOTAL_TIME=$((TOTAL_TIME + RUNTIME))
        echo "Run $i: $RUNTIME ms"
    done
    
    AVG_TIME=$((TOTAL_TIME / REPEATS))
    echo "$INSTANCES;$AVG_TIME" >> $RESULTS_FILE
    echo "Average runtime: $AVG_TIME ms"
done
