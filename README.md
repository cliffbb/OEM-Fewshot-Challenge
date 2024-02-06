<div align="center">
	
## OpenEarthMap Land Cover Mapping Few-Shot Challenge </br> Generalized Few-shot Semantic Segmentation
### Challenge proposed by the [Geoinformatics Team of RIKEN-AIP](https://geoinformatics2018.com/) and co-organized </br> with the [3rd L3D-IVU Workshop](https://sites.google.com/view/l3divu2024/overview) @ CVPR 2024 Conference

<p><img src="docs/assets/img/img2.jpg"></p>
</div>

<div align="center">
	
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
</div>

##
<div align="justify">
<p>
	We are excited to be seeing great ideas from across the globe working hard towards a better understanding of our environment. We look forward to having fun in our quest to obtain more 
	accurate semantic segmentation maps for practical applications of few-shot learning in remote sensing image understanding.	
	Let us come together to push the state-of-the-art generalized few-shot semantic segmentation (GFSS) learning methods to promote research in AI for social good.
</p> 
<p>
	
**Get involved! Check out the following links:** </br>
- Challenge Webpage [https://cliffbb.github.io/OEM-Fewshot-Challenge/](https://cliffbb.github.io/OEM-Fewshot-Challenge/)
- L3D-IVU Workshop @ CVPR 2024 Conference [https://sites.google.com/view/l3divu2024/overview](https://sites.google.com/view/l3divu2024/overview)
- Dataset Download [https://zenodo.org/records/10591939](https://zenodo.org/records/10591939)
- Submission Portal [https://codalab.lisn.upsaclay.fr/competitions/17568?secret_key=94bf7540-3a5f-4bcf-a750-03a06c6d9d23](https://codalab.lisn.upsaclay.fr/competitions/17568?secret_key=94bf7540-3a5f-4bcf-a750-03a06c6d9d23)
</p>
</div>

## Context
<div align="justify">
<p>
	This repository contains the baseline model code for the OpenEarthMap land cover mapping generalized few-shot semantic segmentation challenge, 
	co-organized with the <b>Learning with Limited Labelled Data for Image and Video Understanding</b> Workshop at the <b>CVPR 2024</b> Conference.
	Two phase
</p>
</div>

## Dataset
<div align="justify">
	
This dataset extends the original 8 semantic classes of the [OpenEarthmap](https://open-earth-map.org/) benchmark dataset to 13 classes for **5-shot** generalized few-shot semantic segmentation (GFSS) task with **4 novel classes** and **7 base classes**. It consists of only 408 samples from the original OpenEarthMap dataset. The 408 samples are also split into 258 as *trainset*, 50 as *valset*, and 100 as *testset*. A detailed description of the dataset can be found [here](https://zenodo.org/records/10591939), where it can also be downloaded. 
</div>

## Baseline
<div align="justify">

The PSPNet architecture with EfficientNet-B4 encoder from the [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch?tab=readme-ov-file) GitHub repository is adopted as a baseline network.
The network was pretrained using the *trainset* with the [Catalyst](https://catalyst-team.com/) library. Then, the state-of-the-art framework called [distilled information maximization](https://arxiv.org/abs/2211.14126) 
(DIaM) was adopted to perform the GFSS task. The code in this repository contains only the GFSS portion. As mentioned by the baseline authors, any pretrained model can be used with their framework. 
The code was adopted from [here](https://github.com/sinahmr/DIaM?tab=readme-ov-file). To run the code on the *valset*, simply clone this repository and change your directory to the `OEM-Fewshot-Challenge` folder which contains the code files. Then from a terminal, use the `test.sh` script. The general syntax is:
```bash
bash test.sh 
```
The results of the baseline model on the *valset* are presented below. To reproduce the results, download the pretrained models from [here](https://drive.google.com/file/d/1eLjfUJ2ajAMkJKCsoJr-MGSSzZ-LqDbR/view?usp=sharing). 
Follow the instructions in the **Usage** section, then run the `test.sh` script as explained. 

<table align="center">
    <tr align="center">
        <td>Data</td>
        <td>base mIoU</td> 
	<td>novel mIoU</td> 
	<td>Avg base-nodel mIoU</td>
        <td>Weighted base mIoU</td> 
	<td>Weighted novel mIoU</td>
	<td>Weighted-Sum base-novel mIoU</td>
    </tr>
    <tr align="center">
        <td>Valset</td>
        <td> 29.48 </td> 
	<td> 03.18 </td> 
	<td> 16.33 </td> 
	<td> 11.79 </td> 
	<td> 1.91 </td> 
	<td> 13.70 </td> 
    </tr>
   <tr align="center">
	<td>Testset</td>
        <td> --- </td> 
	<td> --- </td> 
	<td> --- </td> 
	<td> --- </td> 
	<td> --- </td> 
	<td> --- </td> 
    </tr>   
</table>
The weighted mIoUs are calculated using `0.4:0.6 => base:novel`. These weights are derived from the state-of-the-art results presented in the baseline paper.

</div>

## Usage
<div align="justify">


	
</div>

## Citation
<div align="justify">
For any scientific publication using this data, the following paper should be cited:
<pre style="white-space: pre-wrap; white-space: -moz-pre-wrap; white-space: -pre-wrap; white-space: -o-pre-wrap; word-wrap: break-word;">
@InProceedings{Xia_2023_WACV,
    author    = {Xia, Junshi and Yokoya, Naoto and Adriano, Bruno and Broni-Bediako, Clifford},
    title     = {OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {6254-6264}
}
</pre>
</div>

## Acknowledgement
<div align="justify">

We are most grateful to the authors of [DIaM](https://github.com/sinahmr/DIaM?tab=readme-ov-file), [Semantic Segmentation PyTorch](https://github.com/qubvel/segmentation_models.pytorch?tab=readme-ov-file), 
and [Catalyst](https://catalyst-team.com/) from whom parts of the baseline code based on.
</div>








<!-- 
If you want to use the pre-trained models, this step is optional. Our contribution lies in the inference phase and our approach is modular, i.e., it can be applied on top of any segmentation model that is trained on the base classes. We use a simple training scheme by minimizing a standard cross-entropy over base classes. To this end, we have used the train_base.py script and base learner models of BAM (see this issue for more info).



A U-Net architecture with a pre-trained ResNet34 encoder from the pytorch segmentation models library is used for the baselines. The used architecture allows integration of patch-wise metadata information and employs commonly used image data augmentation techniques. It has about 24.4M parameters and it is implemented using the segmentation-models-pytorch library. The results are evaluated with an Intersection Over Union (IoU) metric and a single mIoU is reported (see associated datapaper).

The metadata strategy refers encoding metadata with a shallow MLP and concatenate this encoded information to the U-Net encoder output. The augmentation strategy employs three typical geometrical augmentations (see associated datapaper).

Example of a semantic segmentation of an urban and coastal area in the D076 spatial domain, obtained with the baseline trained model:
	
 A GFSS framework, called distilled information maximization (DIaM), with a PSPNet architecture of EfficientNet-B4 encoder from the PyTorch segmentation models library is presented as a baseline model. The baseline code can be used as a starter code for the challenge submission. To run it, follow the README instructions presented here. After running the code, an output folder ``results`` which contains ``preds`` and ``targets`` folders of the model's segmentation prediction maps and the corresponding targets, respectively, are created. Based on the rules mentioned above, only the ``preds`` folder which contains the predicted segmentation maps in `.png` file format is required for the submission. Please feel free to contact the challenge organizers for any question regarding the baseline code.
A U-Net architecture with a pre-trained ResNet34 encoder from the pytorch segmentation models library is used for the baselines. The used architecture allows integration of patch-wise metadata information and employs commonly used image data augmentation techniques. It has about 24.4M parameters and it is implemented using the segmentation-models-pytorch library. The results are evaluated with an Intersection Over Union (IoU) metric and a single mIoU is reported (see associated datapaper).

The metadata strategy refers encoding metadata with a shallow MLP and concatenate this encoded information to the U-Net encoder output. The augmentation strategy employs three typical geometrical augmentations (see associated datapaper).

Example of a semantic segmentation of an urban and coastal area in the D076 spatial domain, obtained with the baseline trained model:


## &#x1F3AC; 

### :one: Requirements
We used `Python 3.9` in our experiments and the list of packages is available in the `requirements.txt` file. You can install them using `pip install -r requirements.txt`.

### :two: Download data

#### Pre-processed data from drive

We provide the versions of PASCAL VOC 2012 and MS-COCO 2017 used in this work [here](https://etsmtl365-my.sharepoint.com/:u:/g/personal/seyed-mohammadsina_hajimiri_1_ens_etsmtl_ca/Earq9o6KqvJDleNRKqfFZ_cB1AzQCtaZ5g2noh4yjZoecg?e=g1g9t4). You can download the full .zip and directly extract it in the `data/` folder.

#### From scratch

Alternatively, you can prepare the datasets yourself. Here is the structure of the data folder for you to reproduce:

```
data
├── coco
│   ├── annotations
│   ├── train
│   ├── train2014
│   ├── val
│   └── val2014
└── pascal
|   ├── JPEGImages
|   └── SegmentationClassAug
```
**PASCAL**: The JPEG images can be found in the PASCAL-VOC 2012 toolkit to be downloaded at [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [SegmentationClassAug](https://etsmtl365-my.sharepoint.com/:u:/g/personal/seyed-mohammadsina_hajimiri_1_ens_etsmtl_ca/Ef70aWKWEidJoR_NZb131SwB3t7WIHMjJK316qxIu_SPyw?e=CVtNKY) (pre-processed ground-truth masks).

**COCO**: COCO 2014 train images, validation images and annotations can be downloaded at [COCO](https://cocodataset.org/#download). Once this is done, you will have to generate the subfolders `coco/train` and `coco/val` (ground truth masks). Both folders can be generated by executing the python script `data/coco/create_masks.py` (note that this script uses the [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) package):

```
cd data/coco
python create_masks.py
 ```

#### About the train/val splits

The train/val splits are directly provided in `lists/`. How they were obtained is explained at https://github.com/Jia-Research-Lab/PFENet.

### :three: Download pre-trained models

#### Pre-trained backbone and models
We provide the pre-trained backbone and models at https://drive.google.com/file/d/1WuKaJbj3Y3QMq4yw_Tyec-KyTchjSVUG/view?usp=share_link. You can download them and directly extract them at the root of this repo. This will create two folders: `initmodel/` and `model_ckpt/`.

## &#x1F5FA; Overview of the repo

Default configuration files can be found in `config/`. Data are located in `data/`. `lists/` contains the train/val splits for each dataset. All the codes are provided in `src/`. Testing script is located at the root of the repo.

## &#x2699; Training (optional)

If you want to use the pre-trained models, this step is optional. Our contribution lies in the inference phase and our approach is modular, i.e., it can be applied on top of any segmentation model that is trained on the base classes. 
We use a simple training scheme by minimizing a standard cross-entropy over base classes. To this end, we have used the [`train_base.py`](https://github.com/chunbolang/BAM/blob/main/train_base.py) script and base learner models of [BAM](https://github.com/chunbolang/BAM) (see [this issue](https://github.com/sinahmr/DIaM/issues/3) for more info).

## &#x1F9EA; Testing

To test the model, use the `test.sh` script, which its general syntax is:
```bash
bash test.sh {benchmark} {shot} {pi_estimation_strategy} {[gpu_ids]} {log_path}
```
This script tests successively on all folds of the benchmark and reports the results individually. The overall performance is the average over all the folds. Some example commands are presented below, with their description in the comments.

```bash
bash test.sh pascal5i 1 self [0] out.log  # PASCAL-5i benchmark, 1-shot, estimate pi by model's output
bash test.sh pascal10i 5 self [0] out.log  # PASCAL-10i benchmark, 5-shot, estimate pi by model's output
bash test.sh coco20i 5 upperbound [0] out.log  # COCO-20i benchmark, 5-shot, the upperbound model mentioned in the paper
```

If you run out of memory, reduce `batch_size_val` in the config files.

### &#x1F4CA; Results
To reproduce the results, please first download the pre-trained models from [here](https://drive.google.com/file/d/1WuKaJbj3Y3QMq4yw_Tyec-KyTchjSVUG/view?usp=share_link) (also mentioned in the "download pre-trained models" section) and then run the `test.sh` script with different inputs, as explained above.
<table>
    <tr>
        <th colspan="2"></th>
        <th colspan="3">1-Shot</th>
        <th colspan="3">5-Shot</th>
    </tr>
    <tr>
        <th>Benchmark</th>
        <th>Fold</th>
        <th>Base</th> <th>Novel</th> <th>Mean</th>
        <th>Base</th> <th>Novel</th> <th>Mean</th>
    </tr>
    <tr>
        <td rowspan="5"><b>PASCAL-5<sup>i</sup></b></td>
        <td>0</td>
        <td>71.33</td> <td>29.36</td> <td>50.35</td>
        <td>71.06</td> <td>53.72</td> <td>62.39</td>
    </tr>
    <tr>
        <td>1</td>
		<td>69.54</td> <td>46.72</td> <td>58.13</td>
		<td>69.63</td> <td>63.33</td> <td>66.48</td>
    </tr>
    <tr>
        <td>2</td>
		<td>69.10</td> <td>27.07</td> <td>48.09</td>
		<td>69.12</td> <td>54.01</td> <td>61.57</td>
    </tr>
    <tr>
        <td>3</td>
		<td>73.60</td> <td>37.30</td> <td>55.45</td>
		<td>73.60</td> <td>50.19</td> <td>61.90</td>
    </tr>
    <tr>
        <td>mean</td>
		<td>70.89</td> <td>35.11</td> <td>53.00</td>
		<td>70.85</td> <td>55.31</td> <td>63.08</td>
    </tr>
    <tr>
        <td rowspan="5"><b>COCO-20<sup>i</sup></b></td>
        <td>0</td>
		<td>49.01</td> <td>15.89</td> <td>32.45</td>
		<td>48.90</td> <td>24.86</td> <td>36.88</td>
    </tr>
    <tr>
        <td>1</td>
		<td>46.83</td> <td>19.50</td> <td>33.17</td>
		<td>47.10</td> <td>33.94</td> <td>40.52</td>
    </tr>
    <tr>
        <td>2</td>
		<td>48.82</td> <td>16.93</td> <td>32.88</td>
		<td>49.12</td> <td>27.15</td> <td>38.14</td>
    </tr>
    <tr>
        <td>3</td>
		<td>48.45</td> <td>16.57</td> <td>32.51</td>
		<td>48.37</td> <td>28.95</td> <td>38.66</td>
    </tr>
    <tr>
        <td>mean</td>
		<td>48.28</td> <td>17.22</td> <td>32.75</td>
		<td>48.37</td> <td>28.73</td> <td>38.55</td>
    </tr>
    <tr>
        <td rowspan="5"><b>PASCAL-10<sup>i</sup></b></td>
        <td>0</td>
		<td>68.69</td> <td>34.40</td> <td>51.55</td>
		<td>68.49</td> <td>55.94</td> <td>62.22</td>
    </tr>
    <tr>
        <td>1</td>
		<td>71.83</td> <td>28.17</td> <td>50.00</td>
		<td>72.00</td> <td>47.84</td> <td>59.92</td>
    </tr>
    <tr>
        <td>mean</td>
		<td>70.26</td> <td>31.29</td> <td>50.77</td>
		<td>70.25</td> <td>51.89</td> <td>61.07</td>    </tr>
</table>

## &#x1F64F; Acknowledgments

We gratefully thank the authors of [RePRI](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation), [BAM](https://github.com/chunbolang/BAM), [PFENet](https://github.com/Jia-Research-Lab/PFENet), and [PyTorch Semantic Segmentation](https://github.com/hszhao/semseg) from which some parts of our code are inspired.

## &#x1F4DA; Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{hajimiri2023diam,
  title={A Strong Baseline for Generalized Few-Shot Semantic Segmentation},
  author={Hajimiri, Sina and Boudiaf, Malik and Ben Ayed, Ismail and Dolz, Jose},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11269--11278},
  year={2023}
}
``` 
#######################################################
<div align="center">
<p><img src="demo_data/oem_logo.png"></p>
<p>
    <a href="https://github.com/cliffbb/OEM-Lightweight/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-<p>.svg?style=for-the-badge"></a>
    <a href="https://pytorch.org/get-started/previous-versions/"><img src="https://img.shields.io/badge/PYTORCH-1.12+-red?style=for-the-badge&logo=pytorch"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white"></a>
</p>
</div>

# Lightweight Mapping Model
### Overview
___
This is a demo of OpenEarthMap lightweight models searched with
[SparseMask](https://arxiv.org/abs/1904.07642) and
[FasterSeg](https://arxiv.org/abs/1912.10917) neural architecture search methods. 
The models were automatically searched and pretrained on the OpenEarthMap 
[dataset](https://zenodo.org/record/7223446#.Y2Jj1OzP2Ak) 
(using only the training and validation sets).

### OpenEarthMap dataset
___
OpenEarthMap is a benchmark dataset for global high-resolution land cover mapping. 
OpenEarthMap consists of 5000 aerial and satellite images with manually annotated 
8-class land cover labels and 2.2 million segments at a 0.25-0.5m ground 
sampling distance, covering 97 regions from 44 countries across 6 continents. 
OpenEarthMap fosters research, including but not limited to semantic segmentation
and domain adaptation. The project website is https://open-earth-map.org/
```
@inproceedings{xia_2023_openearthmap,
    title = {OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping},
    author = {Junshi Xia and Naoto Yokoya and Bruno Adriano and Clifford Broni-Bediako},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month = {January},
    year = {2023}
}
```

### Lightweight model
___
The lightweight models searched and pretrained on the OpenEarthMap dataset 
can be downloaded as following:

| Method    | Searched architecture   | Pretrained weights           | #Params |  FLOP   |
|:----------|:------------------------|:-----------------------------|:-------:|:-------:|
| SpareMask | [mask_thres_0.001.npy](https://drive.google.com/file/d/1WwE2pIHTb7xGql7xQ9TxeZ1pZmk2JhCl/view?usp=sharing)| [checkpoint_63750.pth.tar](https://drive.google.com/file/d/170o8NNBrrIBJqFdoeYCJoyKHvuub0v2k/view?usp=sharing) | 2.96MB  | 10.45GB |
| FasterSeg | [arch_1.pt](https://drive.google.com/file/d/12oDzi-sDnD_Y4CBONei_g2SZBMZ6cx-2/view?usp=sharing)           | [weights1.pt](https://drive.google.com/file/d/1BgCu1Rz2PvTPJzI_J97hNkr4HvlvI-pE/view?usp=sharing)              | 3.47MB  | 15.43GB |

### Usage
___
* **SparseMask model:** download the [architecture mask](https://drive.google.com/file/d/1WwE2pIHTb7xGql7xQ9TxeZ1pZmk2JhCl/view?usp=sharing) and the [pretrained weights](https://drive.google.com/file/d/170o8NNBrrIBJqFdoeYCJoyKHvuub0v2k/view?usp=sharing)
and put them into folder `models/SparseMask/`.   
Start the evaluation demo as:
```
python eval_oem_lightweight.py \
    --model "sparsemask" \
    --arch "models/SparseMask/mask_thres_0.001.npy" \
    --pretrained_weights "models/SparseMask/checkpoint_63750.pth.tar" \
    --save_image --save_dir "results" 
```   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Or use the Jupyter notebook: `sparsemask_demo.ipynb`.

* **FasterSeg model:** download the [architecture structure](https://drive.google.com/file/d/12oDzi-sDnD_Y4CBONei_g2SZBMZ6cx-2/view?usp=sharing) and the [pretrained weights](https://drive.google.com/file/d/1BgCu1Rz2PvTPJzI_J97hNkr4HvlvI-pE/view?usp=sharing)
and put them into folder `models/FasterSeg/`.   
Start the evaluation demo as:
```
python eval_oem_lightweight.py \
    --model "fasterseg" \
    --arch "models/FasterSeg/arch_1.pt" \
    --pretrained_weights "models/FasterSeg/weights1.pt" \
    --save_image --save_dir "results" 
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Or use the Jupyter notebook `fasterseg_demo.ipynb`.

### Example of predictions
___
* **SparseMask model**   
![](demo_data/sparsemask1.png)    
![](demo_data/sparsemask2.png)
* **FasterSeg model**    
![](demo_data/fasterseg1.png)    
![](demo_data/fasterseg2.png)

### Acknowledgement
___
Automated neural architecture search method code from
* [SparseMask](https://github.com/wuhuikai/SparseMask)
* [FasterSeg](https://github.com/VITA-Group/FasterSeg)

-->




