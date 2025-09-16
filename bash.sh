
TRAIN_SIZE=5000
TEST_SIZE=500
ORACLE_SIZE=20000
SEED=0

G=50000
K=6
T=15
N=50000

SAVE_PATH="/OSYN/Results/"
OPT_DATA_PATH="/OSYN/Opt_data/"

DELTA1=0.01
DELTA2=0.2

A_S_VALUES="[0,  -0.25, -0.5, -0.75, -1, -1.125, -1.25, -1.5, -1.75, -2]"

#Run
!python main_Gen_Quality.py \
        --train_size=$TRAIN_SIZE \
        --test_size=$TEST_SIZE \
        --oracle_size=$ORACLE_SIZE \
        --seed=$SEED \
        --a_s "$A_S_VALUES"\
        --g=$G \
        --k=$K \
        --T=$T \
        --N=$N \
        --save_path=$SAVE_PATH \
        --opt_data_path=$OPT_DATA_PATH \
        --delta1=$DELTA1 \
        --delta2=$DELTA2


