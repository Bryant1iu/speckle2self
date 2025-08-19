1. data中的inVivo的数据给的是Ver.2的
2. 2833: CCA data,  2999: Simulator

python inference.py \
    --data_path data/inVivo/test_data.npy \
    --model_path model_2833.pth \
    --output_path results.npy \
    --visualize

python inference.py \
    --data_path data/simulator/test_data.npy \
    --model_path model_2999.pth \
    --output_path results.npy \
    --visualize
