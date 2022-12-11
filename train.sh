python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/polar_sbd.py --launcher pytorch --validate
