python -m torch.distributed.run --nproc_per_node=1 --master_port 44446 evaluate.py --cfg-path lavis/projects/pnp-vqa/eval/gqa_eval_gpt3_codevqa.yaml
# python -m torch.distributed.run --nproc_per_node=1 --master_port 44446 evaluate.py --cfg-path lavis/projects/pnp-vqa/eval/gqa_eval_gpt3.yaml
