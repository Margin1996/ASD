# Aligning semantic distribution in fusing optical and SAR images for land use classification

This is the core code of our [work](https://www.sciencedirect.com/science/article/pii/S0924271623000977) published in ISPRS J.

This article utilizes the *Semantic Distribution Alignment Loss* to align the semantic features of SAR and Optical modalities, facilitating the fusion of complementary information.

The fundamental concept behind this loss is that two distributions are considered identical if their statistics are the same. 

This loss quantifies the difference between feature distributions by mapping the features into the Reproducing Kernel Hilbert Space (RKHS). 

For further details of this loss, please refer to "[A Kernel Method for the Two-Sample-Problem](https://proceedings.neurips.cc/paper_files/paper/2006/file/e9fb2eda3d9c55a0d89c98d6c54b5b3e-Paper.pdf)."

![Image of work](https://github.com/WHUlwb/ASD/blob/main/network.png)


# Reference

If the project is helpful to you, please consider citing us.
```
@article{li2023aligning,
  title={Aligning semantic distribution in fusing optical and SAR images for land use classification},
  author={Li, Wangbin and Sun, Kaimin and Li, Wenzhuo and Wei, Jinjiang and Miao, Shunxia and Gao, Song and Zhou, Qinhui},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={199},
  pages={272--288},
  year={2023},
  publisher={Elsevier}
}
```
```
@article{gretton2006kernel,
  title={A kernel method for the two-sample-problem},
  author={Gretton, Arthur and Borgwardt, Karsten and Rasch, Malte and Sch{\"o}lkopf, Bernhard and Smola, Alex},
  journal={Advances in neural information processing systems},
  volume={19},
  year={2006}
}
```
