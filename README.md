# Aligning semantic distribution in fusing optical and SAR images for land use classification

This is the core code of our [work](https://www.sciencedirect.com/science/article/pii/S0924271623000977) published in ISPRS J.

This article utilizes the *Semantic Distribution Alignment Loss* to align the semantic features of SAR and Optical modalities, facilitating the fusion of complementary information.

The fundamental concept behind this loss is that two distributions are considered identical if their statistics are the same. 

This loss quantifies the difference between feature distributions by mapping the features into the Reproducing Kernel Hilbert Space (RKHS). 

For further details of this loss, please refer to "[A Kernel Method for the Two-Sample-Problem](https://proceedings.neurips.cc/paper_files/paper/2006/file/e9fb2eda3d9c55a0d89c98d6c54b5b3e-Paper.pdf)."

![Image of FDA](https://github.com/YanchaoYang/FDA/blob/master/demo_images/FDA.png)
