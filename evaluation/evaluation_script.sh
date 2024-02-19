OMP_NUM_THREADS=4 python evaluation_script.py \
    --annFile ./KAIST_annotation.json \
    --rstFile state_of_arts/INSANet_result.txt \
              state_of_arts/CFR_result.txt \
              state_of_arts/GAFF_result.txt \
              state_of_arts/MLPD_result.txt \
              state_of_arts/MBNet_result.txt
    --evalFig KAIST_BENCHMARK.jpg
