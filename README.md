# p3achyjr's Go Bot :)

Hopefully this gets somewhere.

## Building and Running

Assuming inside docker container [docs tbd], run the following commands.

```
mkdir /tmp/p3achygo
mkdir /tmp/shuffler
./sh/build_all_container.sh
```

To run a single process that iteratively runs self-play, trains, and runs eval, do

```
python -m python.rl_loop.train_sp_eval --sp_bin_path=/app/bazel-bin/cc/selfplay/main --eval_bin_path=/app/bazel-bin/cc/eval/main --run_id=${RUN_ID} 2>&1 | tee /tmp/sp_log.txt
```

To run the shuffler, do
```
python -m python.rl_loop.shuffle --bin_path=/app/bazel-bin/cc/shuffler/main --run_id=v1 --local_run_dir=/tmp/shuffler
```

Alternatively, you can run the CC binaries themselves. For eval, do
```
./bazel-bin/cc/eval/main --cur_model_path=${CUR_MODEL_PATH} --cand_model_path=${CAND_MODEL_PATH} --num_games=${NUM_GAMES} --cache_size=${CACHE_SIZE} --cur_n=${CUR_N} --cur_k=${CUR_K} --cand_n=${CAND_N} --cand_k=${CAND_K}
```

## Resources Consulted:

[AlphaGo Fan Paper](https://www.rose-hulman.edu/class/cs/csse413/schedule/day16/MasteringTheGameofGo.pdf)

[AlphaGo Zero Paper](https://www.nature.com/articles/nature24270.epdf)

[KataGo Paper](https://arxiv.org/pdf/1902.10565.pdf)

[Gumbel Policy Scheme for AlphaZero/MuZero](https://openreview.net/pdf?id=bERaNdoegnO)
