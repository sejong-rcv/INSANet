OMP_NUM_THREADS=4 python evaluation_script.py \
    --annFile ./KAIST_annotation.json \
    --rstFile state_of_arts/INSANet_result.txt \
              state_of_arts/MLPD_result.txt \
              state_of_arts/MBNet_result.txt \
              state_of_arts/MSDS-RCNN_result.txt \
              state_of_arts/CIAN_result.txt \
    --evalFig KAIST_BENCHMARK.jpg
