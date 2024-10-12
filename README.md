# InteractTraj

[**Language-Driven Interactive Traffic Trajectory Generation**](https://arxiv.org/pdf/2405.15388)   

![overview](assets/figure1.jpg)


## Environment setup

```bash
git clone -b dev https://github.com/X1a-jk/InteractTraj.git
cd InteractTraj

# Create virtual environment
conda create -n InteractTraj python=3.8
conda activate InteractTraj

# Install other dependency
pip install -r requirements.txt
```

### Download full dataset
We follow the data processing process in [**TrafficGen**](https://github.com/metadriverse/trafficgen/tree/main#cluster-training) and [**LCTGen**](https://github.com/Ariostgx/lctgen):

1. Download from Waymo Open Dataset:

- Register your Google account in: https://waymo.com/open/
- Open the following link with your Google account logged in: https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0
- Download all the proto files from ``waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training_20s``
- Move download files to ``PATH_A``, where you store the raw tf_record files.


2. Data Preprocess
````
python srcs/utils/process_all_data.py PATH_A PATH_B
````
 - Note: ``PATH_B`` is where you store the processed data.

3. Change `_C.DATASET.DATA_PATH` to ``PATH_B`` in `srcs/configs/path_cfg.py`.

### Training with full data
````
chmod +x scripts/train.sh
./scripts/train.sh
````

### Inference and visualization
Modify the path of pre-trained model in srcs/configs/inference.py and the api_key in srcs/utils/inference.py in advance.
````
python srcs/utils/inference.py
````
## TODO
- [x] arxiv paper release
- [x] code release
- [ ] nuPlan dataset preprocess


## Related repositories
We use code in [**TrafficGen**](https://github.com/metadriverse/trafficgen/) and [**LCTGen**](https://github.com/Ariostgx/lctgen). Many thanks to them for the selfless open source of the code and their previous work!



## Citation

```latex
@misc{xia2024languagedriveninteractivetraffictrajectory,
      title={Language-Driven Interactive Traffic Trajectory Generation}, 
      author={Junkai Xia and Chenxin Xu and Qingyao Xu and Chen Xie and Yanfeng Wang and Siheng Chen},
      year={2024},
      eprint={2405.15388},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.15388}, 
}
```



