# Evaluation

## Batch Evaluation

To evaluate the model on a dataset, use the `batch_eval.py` script. It is significantly more efficient in large-scale evaluation compared to `demo.py`, supporting batched inference, multi-GPU inference, torch compilation, and skipping video compositions.

An example of running this script with four GPUs is as follows:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4  batch_eval.py duration_s=8 dataset=vggsound model=small_16k num_workers=8
```

You may need to update the data paths in `config/eval_data/base.yaml`. 
More configuration options can be found in `config/base_config.yaml` and `config/eval_config.yaml`.

## Precomputed Results

Precomputed results for VGGSound, AudioCaps, and MovieGen are available here: https://huggingface.co/datasets/hkchengrex/MMAudio-precomputed-results

## Obtaining Quantitative Metrics

Our evaluation code is available here: https://github.com/hkchengrex/av-benchmark
