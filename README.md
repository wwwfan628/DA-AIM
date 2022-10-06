# DA-AIM: Exploiting Instance-based Mixed Sampling via Auxiliary Source Domain Supervision for Domain-adaptive Action Detection

![](https://img.shields.io/badge/Python-3.9.5-blue.svg)
![](https://img.shields.io/badge/Pytorch-1.8.0-green.svg)
![](https://img.shields.io/badge/PySlowFast-1.0-blueviolet.svg)
![](https://img.shields.io/badge/Tensorboard-2.7.0-orange.svg)

**by Yifan Lu, [Gurkirt Singh](https://gurkirt.github.io/), [Suman Saha](https://sahasuman.bitbucket.io/) 
and [Luc Van Gool](https://scholar.google.de/citations?user=TwMib_QAAAAJ&hl=en)**

We propose a novel domain adaptive action detection approach and a new adaptation protocol that leverages the recent 
advancements in image-level unsupervised domain adaptation (UDA) techniques and handle vagaries of instance-level video
data. Self-training combined with cross-domain mixed sampling has shown remarkable performance gain in semantic 
segmentation in UDA (unsupervised domain adaptation) context. Motivated by this fact, we propose an approach for human 
action detection in videos that transfers knowledge from the source domain (annotated dataset) to the target domain 
(unannotated dataset) using mixed sampling and pseudo-labe- based self-training. The existing UDA techniques follow a 
ClassMix algorithm for semantic segmentation. However, simply adopting ClassMix for action detection does not work, 
mainly because these are two entirely different problems, i.e. pixel-label classification vs. instance-label detection. 
To tackle this, we propose a novel action in stance mixed sampling technique that combines information across domains 
based on action instances instead of action classes. Moreover, we propose a new UDA training protocol that addresses 
the long-tail sample distribution and domain shift problem by using supervision from an auxiliary source domain (ASD). 
For ASD, we propose a new action detection dataset with dense frame-level annotations. We name our proposed framework 
domain-adaptive action instance mixing (DA-AIM). We demonstrate that DA-AIM consistently outperforms prior works on 
challenging domain adaptation benchmarks. 

![overview](pictures/overview.png)

### Demo on IhD-1 Dataset

<div align="center">
  <img src="pictures/IhD1_demo.gif" width="2000px"/>
</div>


### Quantitative Results

<div align=center>
<table style="width:100%">
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="7">AVA-Kinetics → AVA</th>
    <th colspan="4">AVA-Kinetics → IhD-2</th>
  </tr>
  <tr>
    <th>bend/bow</th>
    <th>lie/sleep</th>
    <th>run/jog</th>
    <th>sit</th>
    <th>stand</th>
    <th>walk</th>
    <th>mAP</th>
    <th>touch</th>
    <th>throw</th>
    <th>take a photo</th>
    <th>mAP</th>
  </tr>
  <tr>
    <th>Baseline (Source)</th>
    <td>33.66</td>
    <td>54.82</td>
    <td>56.82</td>
    <td><b><i>73.70</i></b></td>
    <td><b><i>80.56</i></b></td>
    <td><b><i>75.18</i></b></td>
    <td>62.46</td>
    <td>34.12</td>
    <td>32.91</td>
    <td>27.42</td>
    <td>31.48</td>
  </tr>
  <tr>
    <th>Rotation</th>
    <td>25.53</td>
    <td>58.86</td>
    <td>55.05</td>
    <td>72.42</td>
    <td>79.84</td>
    <td>68.49</td>
    <td>60.03</td>
    <td>30.12</td>
    <td>34.58</td>
    <td>25.39</td>
    <td>30.03</td>
  </tr>
  <tr>
    <th>Clip-order</th>
    <td>28.24</td>
    <td>57.38</td>
    <td>56.90</td>
    <td>69.54</td>
    <td>77.10</td>
    <td>74.68</td>
    <td>60.64</td>
    <td>28.28</td>
    <td>32.30</td>
    <td>29.93</td>
    <td>30.17</td>
  </tr>
  <tr>
    <th>GRL</th>
    <td>24.99</td>
    <td>48.41</td>
    <td>59.89</td>
    <td>68.68</td>
    <td>78.79</td>
    <td>71.38</td>
    <td>58.69</td>
    <td>25.79</td>
    <td><b><i>39.71</i></b></td>
    <td>28.90</td>
    <td>31.46</td>
  </tr>
  <tr>
    <th>DA-AIM</th>
    <td><b><i>33.79</i></b></td>
    <td><b><i>59.27</i></b></td>
    <td><b><i>62.16</i></b></td>
    <td>71.67</td>
    <td>79.90</td>
    <td>75.13</td>
    <td><b><i>63.65</i></b></td>
    <td><b><i>34.38</i></b></td>
    <td>35.65</td>
    <td><b><i>39.84</i></b></td>
    <td><b><i>36.62</i></b></td>
  </tr>
  </table>
  </div>


### Qualitative Results

<div align=center>
<table style="width:100%">
  <tr>
    <th>Baseline (Source)</th>
    <th>DA-AIM</th>
  </tr>
  <tr>
    <td><img src="pictures/baseline1.png" width=93% /></td>
    <td><img src="pictures/da_aim1.png" width=100% /></td>
  </tr>
  <tr>
    <td><img src="pictures/baseline2.png" width=93% /></td>
    <td><img src="pictures/da_aim2.png" width=100% /></td>
  </tr>
  </table>
  </div>


The paper can be found here: [[Arxiv]](https://arxiv.org/abs/2209.15439)

In case of interest, please consider citing:

```
@InProceedings{
}
```




## Setup Environment

Python 3.9.5 is used for this project. We recommend setting up a new virtual environment:

```shell
python -m venv ~/venv/daaim
source ~/venv/daaim/bin/activate
```

Clone DA-AIM repository then add this repository to `$PYTHONPATH`.

```
git clone https://github.com/wwwfan628/DA-AIM.git
export PYTHONPATH=/path/to/DA-AIM/slowfast:$PYTHONPATH
```

Build the required environment with:

```shell
cd DA-AIM
python setup.py build develop
```

Before training or evaluation, please download the MiT weights and a pretrained DA-AIM checkpoint using the
following command or directly download them [here]():

```shell
wget https://    # MiT weights
wget https://    # DA-AIM checkpoint
```

## Setup Datasets

Large-scale datasets are reduced because of three reasons:
* action classes needs to be matched to target domain 
* for fair comparison with smaller datasets
* for the sake of time and resource consumption

We provide annotations and the corresponding video frames of the reduced datasets. The numbers contained in the name of 
the dataset suggest the number of action classes, the maximum number of training and validation samples for each action class. For example, 
`ava_6_5000_all` suggests there are `6` action classes, for each action class we have at most `5000` 
training samples and `all` validation samples from the original dataset are kept. Please, download the video frames
and annotations from [here](https://data.vision.ee.ethz.ch/susaha/wacv2023_datasets/) and extract them to `Datasets/Annotations` or
`Datasets/Frames`. Contact author [Suman Saha](mailto:suman.saha@vision.ee.ethz.ch) for more details of downloading.

**ava_6_5000_all:** is reduced from the original **AVA** dataset. We selected `5000` training samples for classes `bend/bow`, 
`lie/sleep`, `run/jog`, `sit`, `stand` and `walk` and kept all validation samples from those classes. 

**kinetics_6_5000_all:** is reduced from the original **AVA-Kinetics** dataset. We selected `5000` training samples for classes `bend/bow`, 
`lie/sleep`, `run/jog`, `sit`, `stand` and `walk` and kept all validation samples from those classes.

You can also use commands to download and extract datasets:
```
wget https://data.vision.ee.ethz.ch/susaha/wacv2023_datasets    # download datasets
tar -C Datasets/Frames -xvf xxx.tar     # extract frames to Datasets/Frames directory
```

The final folder structure should look like this:
```none
Datasets
├── Annotations
│   ├── ava_6_5000_all
│   │   ├── annotations
│   │   │   ├── ava_action_list_v2.2.pbtxt
│   │   │   ├── ava_included_timestamps_v2.2.txt
│   │   │   ├── ava_train_excluded_timestamps_v2.2.csv
│   │   │   ├── ava_val_excluded_timestamps_v2.2.csv
│   │   │   ├── ava_train_v2.2.csv
│   │   │   ├── ava_val_v2.2.csv
│   │   ├── frame_lists
│   │   │   ├── train.csv
│   │   │   ├── val.csv
│   ├── kinetics_6_5000_all
│   │   ├── annotations
│   │   │   ├── kinetics_action_list_v2.2.pbtxt
│   │   │   ├── kinetics_val_excluded_timestamps.csv
│   │   │   ├── kinetics_train.csv
│   │   │   ├── kinetics_val.csv
│   │   ├── frame_lists
│   │   │   ├── train.csv
│   │   │   ├── val.csv
│   ├── ...
├── Frames
│   ├── ava_6_5000_all
│   │   ├── IMG1000
│   │   │   ├──
│   │   │   ├── ...
│   │   ├── IMG1000
│   │   │   ├──
│   │   │   ├── ...
│   │   ├── ...
│   ├── kinetics_6_5000_all
│   │   ├── IMG1000
│   │   │   ├──
│   │   │   ├── ...
│   │   ├── IMG1000
│   │   │   ├──
│   │   │   ├── ...
│   ├── ...
├── ...
```



## Training

**1) Setting configuration yaml files**

Configuration yaml files are located under `./configs/DA-AIM/`. Modify the settings according to your 
requirements. Most of the time, this step can be skipped and necessary settings can be modified in experiment shell files.
More details and explanations of the configuration settings please refer to `./slowfast/config/defaults.py`.

**2) Run experiments**

Experiments can be executed with the following command:
```
python tools/run_net.py --cfg "configs/DA-AIM/KIN2AVA/SLOWFAST_32x2_R50_DA_AIM.yaml" \
        OUTPUT_DIR "/PATH/TO/OUTPUT/DIR" \
        AVA.FRAME_DIR "Datasets/Frames/kinetics_6_5000_all" \
        AVA.FRAME_LIST_DIR "Datasets/Annotations/kinetics_6_5000_all/frame_lists/" \
        AVA.ANNOTATION_DIR "Datasets/Annotations/kinetics_6_5000_all/annotations/" \
        AUX.FRAME_DIR "Datasets/Frames/ava_6_5000_all" \
        AUX.FRAME_LIST_DIR "Datasets/Annotations/ava_6_5000_all/frame_lists/" \
        AUX.ANNOTATION_DIR "Datasets/Annotations/ava_6_5000_all/annotations/" \
        TRAIN.CHECKPOINT_FILE_PATH "/PATH/TO/CKP/FILE"
```
If you skip the first step, please remember to modify the paths to dataset frames and annotations as well as the path 
to the pretrained checkpoint here. Examples of shell scripts are provided under `./experiments`.

## Evaluation

**1) Setting configuration yaml files**

Similarly as in training, configuration settings need to be modified according to requirements. Configuration yaml files are located 
under `./configs/DA-AIM/`.To note is that `TRAIN.ENABLE` should be set as `False` in order to evoke evaluation. As previously mentioned,
this step can be skipped and necessary settings can be modified in shell files.

**2) Perform evaluation**

To perform evaluation, please use the following command:
```
python tools/run_net.py --cfg "configs/DA-AIM/AVA/SLOWFAST_32x2_R50.yaml" \
        TRAIN.ENABLE False \
        TRAIN.AUTO_RESUME True \
        TENSORBOARD.ENABLE False \
        OUTPUT_DIR "/PATH/TO/OUTPUT/DIR" \
        AVA.FRAME_DIR "Datasets/Frames/ava_6_5000_all" \
        AVA.FRAME_LIST_DIR "Datasets/Annotations/ava_6_5000_all/frame_lists/" \
        AVA.ANNOTATION_DIR "Datasets/Annotations/ava_6_5000_all/annotations/" 
```
Also here, attention should be paid to paths to the dataset frames and annotations, if the first step is skipped.

## Demo and Visualization Tools

PySlowFast offers a range of visualization tools. More information at [PySlowFast Visualization Tools](https://github.com/wwwfan628/DA-AIM/blob/main/VISUALIZATION_TOOLS.md). 
Additional visualization tools like plotting mixed samples and confusion matrices, please refer to [DA-AIM Visualization Tools](https://github.com/wwwfan628/DA-AIM/blob/main/DAAIM_VISUALIZATION_TOOLS.md).


## Acknowledgements

This project is based on the following open-source projects. We would like to thank their contributors for implementing
and maintaining their works. Besides, many thanks to labmates Jizhao Xu, Luca Sieber and Rishabh Singh whose previous 
work has lent a helping hand at the beginning of this project. Finally, the filming of IhD-1 dataset is credited to 
Carlos Eduardo Porto de Oliveira's great photography.

* [PySlowFast](https://github.com/facebookresearch/SlowFast)
* [MeanTeacher](https://github.com/CuriousAI/mean-teacher)
* [DACS](https://github.com/vikolss/DACS)
