# Pareto-guided-diffusion-model

This is the official code for the paper [PROUD: PaRetO-gUided Diffusion Model for Multi-objective Generation](https://arxiv.org/abs/2407.04493), accepted by [ECML 2024 Journal Track](https://link.springer.com/article/10.1007/s10994-024-06575-2). 

## Experiments

### CIFAR10
This part adopts [NCSNV2](https://github.com/ermongroup/ncsnv2/tree/master) as the diffusion model backbone. Please follow [NCSNV2](https://github.com/ermongroup/ncsnv2/tree/master) to set up the environment.

Before applying our method, make sure to download the pretrained diffusion model at the [link](https://drive.google.com/drive/folders/1217uhIvLg9ZrYNKOR3XTRFSurt4miQrd?usp=sharing).  

Pareto-guided reverse diffusion:   
`
python main.py --sample --config cifar10.yml -i proud_diversity0.01_1k --guidance proud_diversity --lambda_diversity 0.01 --seed 1
`

### pOAS 
This part adopts the discrete diffusion model in [Gruver et al. (2023)](https://github.com/ngruver/NOS/tree/main) as the backbone for protein sequence generation.


## Acknowledgement
Our codebase references the code from [NCSNV2](https://github.com/ermongroup/ncsnv2/tree/master) and [NOS](https://github.com/ngruver/NOS/tree/main). We thank their authors for open-sourcing their code.


## References

If you find the code/idea useful for your research, please consider citing

```bib
@article{yao2024proud,
  title={PROUD: PaRetO-gUided diffusion model for multi-objective generation},
  author={Yao, Yinghua and Pan, Yuangang and Li, Jing and Tsang, Ivor and Yao, Xin},
  journal={Machine Learning},
  pages={1--28},
  year={2024},
  publisher={Springer}
}
```
