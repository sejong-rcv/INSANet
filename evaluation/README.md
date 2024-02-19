## Evalutation

Anyone can evaluate and visualize the result via code.

We draw all the results of state-of-the-art methods in the figure and the figure represents the log-average miss-rate (LAMR).

For annotation file, only json format is supported.
For result files, json and txt formats are supported. (multiple `--rstFiles` are supported)

Run Example (shell)
```bash
$ python evaluation_script.py \
    --annFile ./KAIST_annotation.json \
    --rstFile state_of_arts/INSANet_result.txt \
              state_of_arts/CFR_result.txt \
              state_of_arts/GAFF_result.txt \
              state_of_arts/MLPD_result.txt \
              state_of_arts/MBNet_result.txt
    --evalFig KAIST_BENCHMARK.jpg

```
![result img](../Doc/figure/state_of_arts.jpg)
