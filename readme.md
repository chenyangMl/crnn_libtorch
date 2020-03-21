## Introduce
Depolying pytorch-trained model with libtorch.

## Requirment
- cmake 3.0
- libtorch-1.4(cpu)
- opencv 4.1.1

## Usage
```shell
cd crnn_libtorch
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="/home/username/libtorch" ..
make -j4
./CrnnDeploy ../src/crnn_model.pt ../src/keys.txt ../src/test.jpg
```
Tips: modify the libtoch path for your lib path.

## Next Working
- Model compression.
