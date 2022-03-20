
# EfficientPyTorch

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
docker run -itd -v [datasets]:/workspace/datasets -v [datasets]:/workspace/EfficientPyTorchPri --gpus all --ipc=host --name efftorch -p 2727:22 [images:id]
docker exec -it [container:id] /bin/bash
```

## Run

```bash
./run_cli.sh examples/classifier_cifar10/prototxt/vggsmall_eagle0.9.prototxt

```
