# Inbreast cancer detection and classification

This repository contains pytorch framework deep learning modules for segementation and classification of Inbreast cancer.

## Classification
This creates models trained only for classification. Models supported by this repo for classification. 

 - Efficient-Net model can be used by specifying the model_name '**efficient_net**' in classification section of config yaml
 - Convnext model can be used by specifying the model_name '**convext**' in classification section of config yaml
 - Resnet34 model can be used by specifying the model_name '**resnet34**' in classification section of config yaml

## Segmentation


This creates models trained for segmentation. Auxilary classifier can. be added to the segmentation model by specifying '**classification : True**'  in segmentation section . 

Architectures supported 

 - Unet model can be used by specifying the model_name '**Unet**' in segmentation section of config 
 - Unet plus plus model can be used by specifying the model_name '**Unetpp**' in segmentation section of config
 - PSPNet model can be used by specifying the model_name '**pspnet**' in segmentation section of config
 - FPN model can be used by specifying the model_name '**fpn**' in segmentation section of config


#### Encoders <a name="encoders"></a>

Encoders supported for the architectures specified above are. Additional you can add auxilary classifier for each of the segmentation model by specifying '**classification**' :True in the segmentation section of the yaml file

Also supports Segfomer network outside the SMP library which can be initialized by specifying the model_name '**segformer**' in segmentation section

Based on SMP library 

<details>
<summary style="margin-left: 25px;">ResNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnet18                        |imagenet / ssl / swsl           |11M                             |
|resnet34                        |imagenet                        |21M                             |
|resnet50                        |imagenet / ssl / swsl           |23M                             |
|resnet101                       |imagenet                        |42M                             |
|resnet152                       |imagenet                        |58M                             |

</div>
</details>

<details>

<summary style="margin-left: 25px;">ResNeXt</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnext50_32x4d                 |imagenet / ssl / swsl           |22M                             |
|resnext101_32x4d                |ssl / swsl                      |42M                             |
|resnext101_32x8d                |imagenet / instagram / ssl / swsl|86M                         |
|resnext101_32x16d               |instagram / ssl / swsl          |191M                            |
|resnext101_32x32d               |instagram                       |466M                            |
|resnext101_32x48d               |instagram                       |826M                            |

</div>
</details>

<details>
<summary style="margin-left: 25px;">ResNeSt</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-resnest14d                 |imagenet                        |8M                              |
|timm-resnest26d                 |imagenet                        |15M                             |
|timm-resnest50d                 |imagenet                        |25M                             |
|timm-resnest101e                |imagenet                        |46M                             |
|timm-resnest200e                |imagenet                        |68M                             |
|timm-resnest269e                |imagenet                        |108M                            |
|timm-resnest50d_4s2x40d         |imagenet                        |28M                             |
|timm-resnest50d_1s4x24d         |imagenet                        |23M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Res2Ne(X)t</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-res2net50_26w_4s           |imagenet                        |23M                             |
|timm-res2net101_26w_4s          |imagenet                        |43M                             |
|timm-res2net50_26w_6s           |imagenet                        |35M                             |
|timm-res2net50_26w_8s           |imagenet                        |46M                             |
|timm-res2net50_48w_2s           |imagenet                        |23M                             |
|timm-res2net50_14w_8s           |imagenet                        |23M                             |
|timm-res2next50                 |imagenet                        |22M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">RegNet(x/y)</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-regnetx_002                |imagenet                        |2M                              |
|timm-regnetx_004                |imagenet                        |4M                              |
|timm-regnetx_006                |imagenet                        |5M                              |
|timm-regnetx_008                |imagenet                        |6M                              |
|timm-regnetx_016                |imagenet                        |8M                              |
|timm-regnetx_032                |imagenet                        |14M                             |
|timm-regnetx_040                |imagenet                        |20M                             |
|timm-regnetx_064                |imagenet                        |24M                             |
|timm-regnetx_080                |imagenet                        |37M                             |
|timm-regnetx_120                |imagenet                        |43M                             |
|timm-regnetx_160                |imagenet                        |52M                             |
|timm-regnetx_320                |imagenet                        |105M                            |
|timm-regnety_002                |imagenet                        |2M                              |
|timm-regnety_004                |imagenet                        |3M                              |
|timm-regnety_006                |imagenet                        |5M                              |
|timm-regnety_008                |imagenet                        |5M                              |
|timm-regnety_016                |imagenet                        |10M                             |
|timm-regnety_032                |imagenet                        |17M                             |
|timm-regnety_040                |imagenet                        |19M                             |
|timm-regnety_064                |imagenet                        |29M                             |
|timm-regnety_080                |imagenet                        |37M                             |
|timm-regnety_120                |imagenet                        |49M                             |
|timm-regnety_160                |imagenet                        |80M                             |
|timm-regnety_320                |imagenet                        |141M                            |

</div>
</details>

<details>
<summary style="margin-left: 25px;">GERNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-gernet_s                   |imagenet                        |6M                              |
|timm-gernet_m                   |imagenet                        |18M                             |
|timm-gernet_l                   |imagenet                        |28M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">SE-Net</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|senet154                        |imagenet                        |113M                            |
|se_resnet50                     |imagenet                        |26M                             |
|se_resnet101                    |imagenet                        |47M                             |
|se_resnet152                    |imagenet                        |64M                             |
|se_resnext50_32x4d              |imagenet                        |25M                             |
|se_resnext101_32x4d             |imagenet                        |46M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">SK-ResNe(X)t</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|timm-skresnet18                 |imagenet                        |11M                             |
|timm-skresnet34                 |imagenet                        |21M                             |
|timm-skresnext50_32x4d          |imagenet                        |25M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">DenseNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|densenet121                     |imagenet                        |6M                              |
|densenet169                     |imagenet                        |12M                             |
|densenet201                     |imagenet                        |18M                             |
|densenet161                     |imagenet                        |26M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Inception</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|inceptionresnetv2               |imagenet /  imagenet+background |54M                             |
|inceptionv4                     |imagenet /  imagenet+background |41M                             |
|xception                        |imagenet                        |22M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">EfficientNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|efficientnet-b0                 |imagenet                        |4M                              |
|efficientnet-b1                 |imagenet                        |6M                              |
|efficientnet-b2                 |imagenet                        |7M                              |
|efficientnet-b3                 |imagenet                        |10M                             |
|efficientnet-b4                 |imagenet                        |17M                             |
|efficientnet-b5                 |imagenet                        |28M                             |
|efficientnet-b6                 |imagenet                        |40M                             |
|efficientnet-b7                 |imagenet                        |63M                             |
|timm-efficientnet-b0            |imagenet / advprop / noisy-student|4M                              |
|timm-efficientnet-b1            |imagenet / advprop / noisy-student|6M                              |
|timm-efficientnet-b2            |imagenet / advprop / noisy-student|7M                              |
|timm-efficientnet-b3            |imagenet / advprop / noisy-student|10M                             |
|timm-efficientnet-b4            |imagenet / advprop / noisy-student|17M                             |
|timm-efficientnet-b5            |imagenet / advprop / noisy-student|28M                             |
|timm-efficientnet-b6            |imagenet / advprop / noisy-student|40M                             |
|timm-efficientnet-b7            |imagenet / advprop / noisy-student|63M                             |
|timm-efficientnet-b8            |imagenet / advprop             |84M                             |
|timm-efficientnet-l2            |noisy-student                   |474M                            |
|timm-efficientnet-lite0         |imagenet                        |4M                              |
|timm-efficientnet-lite1         |imagenet                        |5M                              |
|timm-efficientnet-lite2         |imagenet                        |6M                              |
|timm-efficientnet-lite3         |imagenet                        |8M                             |
|timm-efficientnet-lite4         |imagenet                        |13M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">MobileNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|mobilenet_v2                    |imagenet                        |2M                              |
|timm-mobilenetv3_large_075      |imagenet                        |1.78M                       |
|timm-mobilenetv3_large_100      |imagenet                        |2.97M                       |
|timm-mobilenetv3_large_minimal_100|imagenet                        |1.41M                       |
|timm-mobilenetv3_small_075      |imagenet                        |0.57M                        |
|timm-mobilenetv3_small_100      |imagenet                        |0.93M                       |
|timm-mobilenetv3_small_minimal_100|imagenet                        |0.43M                       |

</div>
</details>

<details>
<summary style="margin-left: 25px;">DPN</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|dpn68                           |imagenet                        |11M                             |
|dpn68b                          |imagenet+5k                     |11M                             |
|dpn92                           |imagenet+5k                     |34M                             |
|dpn98                           |imagenet                        |58M                             |
|dpn107                          |imagenet+5k                     |84M                             |
|dpn131                          |imagenet                        |76M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">VGG</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|vgg11                           |imagenet                        |9M                              |
|vgg11_bn                        |imagenet                        |9M                              |
|vgg13                           |imagenet                        |9M                              |
|vgg13_bn                        |imagenet                        |9M                              |
|vgg16                           |imagenet                        |14M                             |
|vgg16_bn                        |imagenet                        |14M                             |
|vgg19                           |imagenet                        |20M                             |
|vgg19_bn                        |imagenet                        |20M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Mix Vision Transformer</summary>
<div style="margin-left: 25px;">

Backbone from SegFormer pretrained on Imagenet! Can be used with other decoders from package, you can combine Mix Vision Transformer with Unet, FPN and others!

Limitations:  

   - encoder is **not** supported by Linknet, Unet++
   - encoder is supported by FPN only for encoder **depth = 5**

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|mit_b0                          |imagenet                        |3M                              |
|mit_b1                          |imagenet                        |13M                             |
|mit_b2                          |imagenet                        |24M                             |
|mit_b3                          |imagenet                        |44M                             |
|mit_b4                          |imagenet                        |60M                             |
|mit_b5                          |imagenet                        |81M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">MobileOne</summary>
<div style="margin-left: 25px;">

Apple's "sub-one-ms" Backbone pretrained on Imagenet! Can be used with all decoders.

Note: In the official github repo the s0 variant has additional num_conv_branches, leading to more params than s1.

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|mobileone_s0                    |imagenet                        |4.6M                              |
|mobileone_s1                    |imagenet                        |4.0M                              |
|mobileone_s2                    |imagenet                        |6.5M                              |
|mobileone_s3                    |imagenet                        |8.8M                              |
|mobileone_s4                    |imagenet                        |13.6M                             |

</div>
</details>

---

## Commands 
 - Install the required libraries by `pip install -r requirements.txt`
 - Run the code `python main.py --config PATH_TO_CONFIG `
 - Sample config file is shown config folder 
 - Change train boolen in model section for testing set it **False**

---
### TODO
 - [ ] Aux classifiers for SMP architecutures (only supported for segformer currently)
 - [ ] Parallelize models 
 - [ ] Add hybrid models
 - [ ] Add region proposal
