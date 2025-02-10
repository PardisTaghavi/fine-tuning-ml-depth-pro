This repo aims fine-tuning Depth Pro model. you can find original repo HERE https://github.com/apple/ml-depth-pro?tab=readme-ov-file




## Getting Started

```bash
conda create -n depth-pro -y python=3.9
conda activate depth-pro

pip install -e .
```

To download pretrained checkpoints follow the code snippet below:
```bash
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.
```

data - 186
```
gdown https://drive.google.com/file/d/1O36VsMEFS20Vmd4Mt9kul0zuLqSHUTtN/view?usp=sharing
```

data - 186/1860
```
gdown https://drive.google.com/file/d/1_1wQAFVUbXz8p0sVIJRvpFY4hFTMy-N9/view?usp=drive_link
```

fine-tuning
```
torchrun --nproc_per_node=2 finetune_teacher_depth.py --batch_size 16 --epochs 10 --data_dir /home/avalocal/thesis23/KD/ml-depth-pro/train_few_shot_16 --lr 1e-5
```
Cityscapes
Model| lr   | d1 | d2 | d3 | abs_rel | sq_rel | rmse | rmse_log | log10 | silog |
|----|------|----|----|----|----------|--------|-----|----------|-------|-------|
|0shot | N/A |0.743 |0.917|0.963|0.206|0.039|0.116|0.415|0.087|0.413
|Few-shot(186) | 1e-5 |0.90|0.97|0.98|0.09|1.36|6.9|0.21|0.04|0.21












