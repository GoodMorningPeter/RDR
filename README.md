<h1 align="center">Revive Re-weighting in Imbalanced Learning by Density Ratio Estimation</h1>

<p align="center">
<a href="https://openreview.net/pdf?id=vx4NgdyyVG"><img src="https://img.shields.io/badge/OpenReview-NeurIPS'24-blue" alt="Paper"></a>
<a href="https://github.com/GoodMorningPeter/RDR"><img src="https://img.shields.io/badge/Github-RDR-brightgreen?logo=github" alt="Github"></a>
</p>

by Jiaan Luo*, Feng Hong*, Jiangchao Yao, Bo Han, Ya Zhang, Yanfeng Wang at SJTU, Shanghai AI Lab, and HKBU.


## Citation

```
@inproceedings{
luo2024revive,
title={Revive Re-weighting in Imbalanced Learning by Density Ratio Estimation},
author={Luo Jiaan and Hong Feng and Yao Jiangchao and Han Bo and Zhang Ya and Wang Yanfeng},
booktitle={NeurIPS},
year={2024}
}
```

## Environment

The project has been tested under the following environment settings:
- OS: Ubuntu 18.04.5 LTS
- GPU: NVIDIA GeForce RTX 3090
- CUDA: 11.7
- Cudatoolkit: 11.0.221
- Python: 3.8.18
- PyTorch: 1.13.1


## Content

**Structure**

- ```./utils```: data augmentation, model management and evaluation
- ```./models```: backbone models
- ```./data```: datasets (automatically downloaded)
- ```main.py```: main function script
- ```imbalance_cifar.py```: custom datasets script

**Run**

```shell
python main.py --gpu 0 --imb_type exp --dataset cifar10 --imb_factor 0.01
```

**Evaluate**

```shell
python main.py --gpu 0 --evaluate --resume path/to/saved/model --dataset cifar10 --imb_factor 0.01
```


**Implement Other Datasets**

- Add raw data to ```./[_data_name]```.
- Create a imbalanced version of the dataset in ```./data```. (Optional)
- Create a dataloader in ```./utils```.


## Contact
If you have any problem with this code, please feel free to contact **luojiaan@sjtu.edu.cn**.
