#!/bin/bash
#SBATCH --job-name=M3L15                         # 作業名稱
#SBATCH --partition=GPU-MEDIUM                   # 分區名稱，請確認是否正確
#SBATCH --mail-type=END,FAIL                     # 在作業結束或失敗時發送郵件
#SBATCH --mail-user=zw023@ie.cuhk.edu.hk         # 你的郵箱地址
#SBATCH --output=M3L15_%j.log                   # 輸出日誌文件，%j為作業ID
#SBATCH --gres=gpu:1                             # 請求1個GPU資源

# 激活conda環境
eval "$(conda shell.bash hook)"
conda activate my_env

# 切換到工作目錄
cd ~/link-enhancement

# 執行GaPFL算法，遍歷不同的H值
for pe in 0.075 0.1 0.125 0.15 ; do
    python3 simple_launcher.py --Ka=50 --pe=$pe --L=15 --sic=1 --M=3 --ctype=B --num_exp=3 --toPrint=0
done