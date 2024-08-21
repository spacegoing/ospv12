echo $AIHC_JOB_NAME
GPUS_PER_NODE=8
if [ $WORLD_SIZE -eq 1 ]; then
    echo "$AIHC_JOB_NAME-master-0 slots=$GPUS_PER_NODE" >> hostfile
else
    echo "$AIHC_JOB_NAME-master-0 slots=$GPUS_PER_NODE" >> hostfile
    for ((index=0; index<$WORLD_SIZE-1; index++))
    do
        echo "$AIHC_JOB_NAME-worker-${index} slots=$GPUS_PER_NODE" >> hostfile
    done
fi
