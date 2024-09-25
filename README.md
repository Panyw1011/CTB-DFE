# 基于复合三分支和深度特征编码的实时语义分割算法
Real-Time Semantic Segmentation Algorithm based on Composite Three-Branch and Deep Feature Encoding

## Installation
```
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
cd SegFormer && pip install -e . --user
```

## Training
```
# Single-gpu training
python tools/train.py local_configs/ctbdfe/B0/b0.512x512.ade.160k.py 

# Multi-gpu training
./tools/dist_train.sh local_configs/ctbdfe/B0/b0.512x512.ade.160k.py <GPU_NUM>
```
