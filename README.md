# PanNet

Pansharpenning method pytorch implementation

Pretrained model is provided

Based on implementation: https://github.com/xyc19970716/Deep-Learning-PanSharpening/tree/main

Paper link: [https://www.mdpi.com/2072-4292/8/7/594](https://xueyangfu.github.io/paper/2017/iccv/YangFuetal2017.pdf)

# Dataset

The GaoFen-2 and WorldView-3 dataset download links can be found in https://github.com/liangjiandeng/PanCollection

# Torch Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 4, 256, 256]         --
|    └─ConvTranspose2d: 2-1              [-1, 4, 256, 256]         1,028
├─Sequential: 1-2                        [-1, 32, 256, 256]        --
|    └─Conv2d: 2-2                       [-1, 32, 256, 256]        1,472
|    └─ReLU: 2-3                         [-1, 32, 256, 256]        --
├─Sequential: 1-3                        [-1, 4, 256, 256]         --
|    └─Sequential: 2-4                   [-1, 32, 256, 256]        --
|    |    └─Residual_Block: 3-1          [-1, 32, 256, 256]        18,624
|    |    └─Residual_Block: 3-2          [-1, 32, 256, 256]        18,624
|    |    └─Residual_Block: 3-3          [-1, 32, 256, 256]        18,624
|    |    └─Residual_Block: 3-4          [-1, 32, 256, 256]        18,624
|    └─Conv2d: 2-5                       [-1, 4, 256, 256]         1,156
==========================================================================================
Total params: 78,152
Trainable params: 78,152
Non-trainable params: 0
Total mult-adds (G): 5.07
==========================================================================================
Input size (MB): 0.25
Forward/backward pass size (MB): 276.00
Params size (MB): 0.30
Estimated Total Size (MB): 276.55
==========================================================================================
```
# Quantitative Results
## GaoFen-2

![alt text](https://github.com/nickdndndn/PanNet/blob/main/results/Figures_GF2.png?raw=true)


## WorldView-3

![alt text](https://github.com/nickdndndn/PanNet/blob/main/results/Figures_WV3.png?raw=true)

# Qualitative Results
## GaoFen-2

![alt text](https://github.com/nickdndndn/PanNet/blob/main/results/Images_GF2.png?raw=true)

## WorldView-3

![alt text](https://github.com/nickdndndn/PanNet/blob/main/results/Images_WV3.png?raw=true)

