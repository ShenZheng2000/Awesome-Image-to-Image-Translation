[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This is an repository that contains the resources for image-to-image translation (I2I) research. 

# Papers

## AAAI 2023
* MIDMs: Matching Interleaved Diffusion Models for Exemplar-based Image Translation [[pdf](https://arxiv.org/abs/2209.11047)] [[code pending](https://github.com/KU-CVLAB/MIDMs)]
  * a diffusion-based matching-and-generation framework that interleaves cross-domain matching and diffusion in the latent space for I2I

## NeurIPS 2022
* EGSDE: Unpaired Image-to-Image Translation via Energy-Guided Stochastic Differential Equations [[pdf](https://arxiv.org/abs/2207.06635)] [[code](https://github.com/ML-GSAI/EGSDE)]
  * an energy-guided stochastic differential equations that utilizes an energy function pretrained on source and target domains to guide the SDE inference for unpaired I2I

* Unsupervised Image-to-Image Translation with Density Changing Regularization [[pdf](https://arxiv.org/abs/2204.03641)] [[code](https://github.com/Mid-Push/Decent)]
  * an unsupervised I2I model based on a density changing assumption that we should match image patches of high probability density for different domains. 

## ECCV 2022
* Multi-Curve Translator for High-Resolution Photorealistic Image Translation [[pdf](https://arxiv.org/abs/2203.07756)] [[code](https://github.com/IDKiro/MCT)]
  * a Multi-Curve Translator which predicts both the individual pixels and the neighor pixels for high-resolution I2I.

* ManiFest: Manifold Deformation for Few-shot Image Translation [[pdf](https://arxiv.org/abs/2111.13681)] [[code](https://github.com/astra-vision/ManiFest)]
  * a few-shot image translation model that learns a context-aware representation of a target domain using a style manifold between source and proxy anchor domains.

* Vector Quantized Image-to-Image Translation [[pdf](https://arxiv.org/abs/2207.13286)] [[code](https://github.com/cyj407/VQ-I2I)]
  * A I2I framework based on vector quantized content representation 

* Unpaired Image Translation via Vector Symbolic Architectures [[pdf](https://arxiv.org/abs/2209.02686)] [[code](https://github.com/facebookresearch/vsait)]
  * a I2I framework based on Vector Symbolic Architectures which defines algebraic operations in a hypervector space. 

* VecGAN: Image-to-Image Translation with Interpretable Latent Directions [[pdf](https://arxiv.org/abs/2207.03411)] [**NO CODE**]
  * a I2I framework with interpretable latent directions using latent space factorization and controllable strength of change. 

* Bi-level Feature Alignment for Versatile Image Translation and Manipulation [[pdf](https://arxiv.org/abs/2107.03021)] [[code](https://github.com/fnzhan/RABIT)]
  * a I2I framework using a bi-level feature alignment strategy that adopts a top-k operation to rank block-wise features and dense attention between block features to reduce memory cost.

## CVPR 2022
* Exploring Patch-Wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks [[pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Jung_Exploring_Patch-Wise_Semantic_Relation_for_Contrastive_Learning_in_Image-to-Image_Translation_CVPR_2022_paper.html)] [[code](https://github.com/jcy132/Hneg_SRC)]
  * a I2I framework based on semantic relation consistency and regularization along with the decoupled contrastive learning

* Alleviating Semantics Distortion in Unsupervised Low-Level Image-to-Image Translation via Structure Consistency Constraint [[pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Alleviating_Semantics_Distortion_in_Unsupervised_Low-Level_Image-to-Image_Translation_via_Structure_CVPR_2022_paper.html)] [[code](https://github.com/cr-gjx/scc)]
  * a Structure Consistency Constraint that reduces the randomness of color transformation in I2I.

* A Style-Aware Discriminator for Controllable Image Translation [[pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_A_Style-Aware_Discriminator_for_Controllable_Image_Translation_CVPR_2022_paper.html)] [[code](https://github.com/kunheek/style-aware-discriminator)]
  * a style-aware discriminator that acts as both the critic and the style encoder to provide conditions for the generator in I2I. 

* Wavelet Knowledge Distillation: Towards Efficient Image-to-Image Translation [[pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Wavelet_Knowledge_Distillation_Towards_Efficient_Image-to-Image_Translation_CVPR_2022_paper.html)] [**NO CODE**]
  * a I2I method based on high frequency bands distillation from discrete wavelet transformation.

* **InstaFormer: Instance-Aware Image-to-Image Translation With Transformer** [[pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_InstaFormer_Instance-Aware_Image-to-Image_Translation_With_Transformer_CVPR_2022_paper.html)] [[code](https://github.com/KU-CVLAB/InstaFormer)]
  * a transformer-based architecture with with adaptive instance normalization for instance-aware I2I. 

* Maximum Spatial Perturbation Consistency for Unpaired Image-to-Image Translation [[pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Maximum_Spatial_Perturbation_Consistency_for_Unpaired_Image-to-Image_Translation_CVPR_2022_paper.html)] [[code](https://github.com/batmanlab/mspc)]
  * a universal regularization technique for I2I called maximum spatial perturbation consistency which enforces the spatial perturbation function and translation operator to be commutative.

* FlexIT: Towards Flexible Semantic Image Translation [[pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Couairon_FlexIT_Towards_Flexible_Semantic_Image_Translation_CVPR_2022_paper.html)] [[code](https://github.com/facebookresearch/semanticimagetranslation)]
  * a semantic image translation method based on autoencoder latent space and multi-modal embedding space

* Self-Supervised Dense Consistency Regularization for Image-to-Image Translation [[pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Ko_Self-Supervised_Dense_Consistency_Regularization_for_Image-to-Image_Translation_CVPR_2022_paper.html)] [**NO CODE**]
  * an auxiliary self-supervision loss with dense consistency regularization for I2I. 

* Unsupervised Image-to-Image Translation With Generative Prior [[pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Unsupervised_Image-to-Image_Translation_With_Generative_Prior_CVPR_2022_paper.html)] [[code](https://github.com/williamyang1991/gp-unit)]
  * a I2I framework that uses the generative prior from GANs to learn rich content correspondences across various domains

* QS-Attn: Query-Selected Attention for Contrastive Learning in I2I Translation [[pdf](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_QS-Attn_Query-Selected_Attention_for_Contrastive_Learning_in_I2I_Translation_CVPR_2022_paper.html)] [[code](https://github.com/sapphire497/query-selected-attention)]
  * a I2I framework based on a query-selected attention module, which compares feature distances in the source domain and select queries acc. to the measurement of signficance. 
 
## AAAI 2022
* OA-FSUI2IT: A Novel Few-Shot Cross Domain Object Detection Framework with Object-Aware Few-Shot Unsupervised Image-to-Image Translation [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/20253)] [**NO CODE**]
  * an Object-Aware Few-Shot Image Translation framework for few-shot cross domain object detection 

* Style-Guided and Disentangled Representation for Robust Image-to-Image Translation [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/19924)] [**NO CODE**]
  * a I2I framework with a style-guided disriminator using flexible decision boundary and independent domain attributes

# Surveys

* Image-to-image translation: Methods and applications. 2021 [[pdf](https://arxiv.org/abs/2101.08629)]
  * lightweight network design for better efficiency
  * generalize to cross-modality tasks (e.g., NLP, speech)

* Deep Generative Adversarial Networks for Image-to-Image Translation: A Review paper. 2020 [[pdf](https://www.mdpi.com/2073-8994/12/10/1705)] 
  * Solve mode collapse
  * More realistic evaluation metrics
  * More image diversity
  * Deep Reinforcement Learning
  * 3D image-to-image translation
  * 3D datasets
  * Cybersecurity applications 

* An Overview of Image-to-Image Translation Using Generative Adversarial Networks [[pdf](https://link.springer.com/chapter/10.1007/978-3-030-68780-9_31)] 
  * Combine GAN with other methods (e.g., VAE) to stabilize training
  * GAN compression for lightweight design
  * Transfer other methods (e.g., SR, Attention, OT) to I2IT
  * Remove unnecessary components
  * Extend to video
  
* Unsupervised Image-to-Image Translation: A Review [[pdf](https://www.mdpi.com/1424-8220/22/21/8540)] 

* Applications of I2I to rainy days
  * Domain Bridge for Unpaired Image-to-Image Translation and Unsupervised Domain Adaptation [[pdf](https://openaccess.thecvf.com/content_WACV_2020/papers/Pizzati_Domain_Bridge_for_Unpaired_Image-to-Image_Translation_and_Unsupervised_Domain_Adaptation_WACV_2020_paper.pdf)]
  
  * Closing the Loop: Joint Rain Generation and Removal via Disentangled Image Translation [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Ye_Closing_the_Loop_Joint_Rain_Generation_and_Removal_via_Disentangled_CVPR_2021_paper.pdf)]
  
  * From Rain Generation to Rain Removal [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_From_Rain_Generation_to_Rain_Removal_CVPR_2021_paper.pdf)]
  
  * DerainCycleGAN: Rain Attentive CycleGAN for Single Image Deraining and Rainmaking [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9420312&casa_token=U2REa5Ff6vgAAAAA:9wrW8ZZKIYeeJdGlHzARzdpSRq_B0lbQVQ350rXic-4hpSWOgrJSloRdgoq7OT3lzG-k5BXk)]
  
  * Close the Loop: A Unified Bottom-up and Top-down Paradigm for Joint Image Deraining and Segmentation [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/20033)]
  
# Datasets
* [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

* [RaFD](http://www.rafd.nl/)

* [CMP Facades](https://cmp.felk.cvut.cz/~tylecr1/facade/)

* [Facescrub](http://vintage.winklerbros.net/facescrub.html)

* [Cityscapes](https://www.cityscapes-dataset.com/)

* [Helen Face](http://www.ifp.illinois.edu/~vuongle2/helen/)

* [CartoonSet](https://google.github.io/cartoonset/)

* [ImageNet](https://www.image-net.org/)

* [UT-Zap50K](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/)

# Metrics
* Amazon Mechanical Turk (AMT)

* Peak Signal-to-Noise Ratio (PSNR) ↑

* Structural Similarity Index Measure (SSIM) ↑

* Inception Score (IS) ↑

* Fréchet Inception Distance (FID) ↓

* Kernel Inception Distance (KID) ↓

* Perceptual Distance (PD) ↓

* Learned Perceptual Image Patch Similarity (LPIPS) ↓ 

* FCN ↑

* Density and Coverage (DC) ↑

# Resources
* [awesome-image-translation](https://github.com/weihaox/awesome-image-translation)

* [awesome-image-synthesis](https://github.com/Victarry/awesome-image-synthesis)
