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
VAR_DIR="$HOME_DIR/c_omp"

W=1.5
N=2000
E=0.000000000001

RESULTS_FILE="c_omp_multicore_results.csv"
echo "nodes;time_ms" > $RESULTS_FILE
REPEATS=100 

for INSTANCES in {1..4}; do
    echo "$INSTANCES nodes"
    TOTAL_TIME=0
    
    for ((i=1; i<=REPEATS; i++)); do
        START_TIME=$(date +%s%3N) 

        "$VAR_DIR/build/sor"  \
            -t $INSTANCES \
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