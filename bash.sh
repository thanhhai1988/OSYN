
TRAIN_SIZE=5000
TEST_SIZE=500
ORACLE_SIZE=20000
SEED=0

G=50000
K=3
T=10
N=50000

SAVE_PATH="/content/OSYN/Results/"
OPT_DATA_PATH="/content/OSYN/Opt_data/"

DELTA1=0.01
DELTA2=0.2

# ---- Ví dụ sweep qua nhiều giá trị a_s ----
A_S_VALUES=("-0.25" "-0.5" "-0.75")

# ---- Loop chạy nhiều thí nghiệm ----
python main.py \
        --train_size=$TRAIN_SIZE \
        --test_size=$TEST_SIZE \
        --oracle_size=$ORACLE_SIZE \
        --seed=$SEED \
        --a_s $A_S \
        --g=$G \
        --k=$K \
        --T=$T \
        --N=$N \
        --save_path=$SAVE_PATH \
        --opt_data_path=$OPT_DATA_PATH \
        --delta1=$DELTA1 \
        --delta2=$DELTA2