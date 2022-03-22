
# EfficientPyTorch

- [x] This is an implementation of channel pruning using the latest `torch.FX` feature.
- [x] This is also a reimplementation of the paper `EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning`.
  - [x] A [modification](https://github.com/hustzxd/EagleEyeEFF/blob/c2909ae27e4f5e62068ab60612bf463adbcba136/examples/classifier_imagenet/prototxt/resnet50_eagle0.5w0a0.prototxt#L27) to random search to make the search process more stable.
## References
1. https://github.com/anonymous47823493/EagleEye
2. https://github.com/IntelLabs/distiller

## Install
```bash
git clone https://github.com/hustzxd/EagleEyeEFF.git

cd EagleEyeEFF
# ./setup.sh
source setup.sh
pip install -r requirements.txt
```

## Docker
```bash
docker build docker/ -t efftorch:torch1.10.0
docker run -itd -v [datasets]:/workspace/datasets -v [repo]:/workspace/EagleEyeEFF --gpus all --ipc=host --name efftorch [images:id]
docker exec -it [container:id] /bin/bash
```

## Run

```bash
./run_cli.sh examples/classifier_cifar10/prototxt/vggsmall_eagle0.9.prototxt
./run_cli.sh examples/classifier_imagenet/prototxt/resnet50_eagle0.5w0a0.prototxt
./run_cli.sh examples/classifier_imagenet/prototxt/mobilenetv1_eagle0.5w0a0.prototxt
```
