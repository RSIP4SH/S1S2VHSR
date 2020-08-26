# S1S2THRS
Multi-sensor and Multi-scale data fusion for land cover mapping through Convolutional Neural Networks (CNN). 

This repository supports a paper we have submitted to IEEE JSTARS. The study assess the fusion of Sentinel-1 and Sentinel-2 satellite image time series in addition to a Very High Spatial Resolution (VHRS) SPOT image for land cover mapping via a 3 branch CNN architecture. The study was carried out on Reunion island.

## Prerequisites

The code relies on Pyhton 3.7.6. The CNN models were implemented with Tensorflow 2. 

## Examples 

### Running the main architecture model

- fusion of S1, S2 and SPOT

```
python main.py s1_path s2_path ms_path pan_path gt_path
```

- See help for descriptions `python main.py -h/--help`

### Running ablation models 

- available cases: S1 and S2, S2 and SPOT, S1, S2 or SPOT)

- example for S1 and S2 `python main.py s1_path s2_path ms_path pan_path gt_path -s s1 s2`

- example for S2 and SPOT `python main.py s1_path s2_path ms_path pan_path gt_path -s s2 spot`

### Change some hyperparameters

- See help for all available options

- batch size: `-bs` default value 256

- learning rate: `-lr`default value 0.0001

- number of epochs: `-ep` default value 1000
