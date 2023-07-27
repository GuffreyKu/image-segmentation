# image-segmentation


Environment:

1. Install Xcode : xcode-select --install
2. Install Homebrew : /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
3. Install miniforge : brew install miniforge
4. conda create -n torch python=3.10
5. conda activate torch

Python tool :

1. pip install torch torchvision torchaudio
2. pip install coremltools
3. conda install -c conda-forge opencv
4. pip install pandas
5. pip install imgaug
6. pip install torch-optimizer

How to training :

1. download dataset : https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset
2. put in data to "train_python/dataset" folder.
3. run data_prep.ipynb
4. python trainer.py

How to convert model :
1. modify model path
2. python pt2ct.py

How to inference(need to check model path ):

1. inference on ANE : python inference_ane.py
2. inference on GPU : python inference_gpu.py
3. inference on GPU with C++(need to check DCMAKE_PREFIX_PATH):
    
    * cd inference_cpp
    * mkdir build & cd build
    * cmake -DCMAKE_PREFIX_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/torch/lib/python3.10/site-packages/torch ..
    * cmake --build . --config Release
    * make -j4
    * ./human_seg


Good Luck