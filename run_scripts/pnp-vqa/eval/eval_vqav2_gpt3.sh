python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/pnp-vqa/eval/vqav2_eval_gpt3_codevqa.yaml
python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/pnp-vqa/eval/vqav2_eval_gpt3.yaml
