#!/bin/bash
#SBATCH --job-name=c_omp_multicore_test
#SBATCH --output=c_omp_multicore_results.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1      
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4     
#SBATCH --mem=2G
#SBATCH --partition=tornado

HOME_DIR="$HOME/parallel_sor"
LINSYS="$HOME_DIR/linsys"
OUTS="$HOME_DIR/outs"
C_OMP_DIR="$HOME_DIR/c_omp"

W=1.5
N=300
E=0.000000000001

RESULTS_FILE="c_omp_multicore_results.csv"
echo "nodes;time_ms" > $RESULTS_FILE

for INSTANCES in {1..4}; do
    echo "$INSTANCES nodes"
    
    START_TIME=$(date +%s%3N) 

    "$C_OMP_DIR/build/sor"  \
        -t $INSTANCES \
        -c "$LINSYS/${N}.txt" \
        -o "$OUTS/c_omp_${N}_${INSTANCES}.txt" \
        -n $N -e $E -w $W

    END_TIME=$(date +%s%3N) 
    RUNTIME=$((END_TIME - START_TIME))

    echo "$INSTANCES;$RUNTIME" >> $RESULTS_FILE
    echo "runtime: $RUNTIME ms"
done
