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

data
```
gdown https://drive.google.com/file/d/1O36VsMEFS20Vmd4Mt9kul0zuLqSHUTtN/view?usp=sharing
```

fine-tuning
```
torchrun --nproc_per_node=2 finetune_teacher_depth.py --batch_size 16 --epochs 10 --data_dir /home/avalocal/thesis23/KD/ml-depth-pro/train_few_shot_16
```



