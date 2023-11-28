# SFA-Net

- This repository presents an architecture for urban scene segmentation in high-resolution remote sensing images, with support for both training and testing.


## Install

```
conda create -n noname python=3.8
conda activate noname
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```


## Folder Structure

Prepare the following folders to organize this repo:
```none
SFA-Net
├── network
├── config
├── tools
├── model_weights (save the model weights)
├── fig_results (save the masks predicted)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
│   ├── uavid
│   │   ├── uavid_train (original)
│   │   ├── uavid_val (original)
│   │   ├── uavid_test (original)
│   │   ├── uavid_train_val (Merge uavid_train and uavid_val)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── train_val (processed)
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
```


## Data Preprocessing

Download Datasets
- [ISPRS Vaihingen, Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
- [UAVid](https://uavid.nl/)
- [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)

Configure the folder as shown in 'Folder Structure' above.

**UAVid**

```
python tools/uavid_patch_split.py --input-dir "data/uavid/uavid_train_val" --output-img-dir "data/uavid/train_val/images" --output-mask-dir "data/uavid/train_val/masks" --mode "train" --split-size-h 1024 --split-size-w 1024 --stride-h 1024 --stride-w 1024
```
```
python tools/uavid_patch_split.py --input-dir "data/uavid/uavid_train" --output-img-dir "data/uavid/train/images" --output-mask-dir "data/uavid/train/masks" --mode 'train' --split-size-h 1024 --split-size-w 1024 --stride-h 1024 --stride-w 1024
```
```
python tools/uavid_patch_split.py --input-dir "data/uavid/uavid_val" --output-img-dir "data/uavid/val/images" --output-mask-dir "data/uavid/val/masks" --mode 'val' --split-size-h 1024 --split-size-w 1024 --stride-h 1024 --stride-w 1024
```

**Vaihingen**

The [paper]() contains the identity splits for all datasets.


- Using 3 zip files: ISPRS_semantic_labeling_Vaihingen.zip, ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip, ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip
- 'gts_for_participants' folder of ISPRS_semantic_labeling_Vaihingen.zip --> train_masks
- Files in the 'top' folder of ISPRS_semantic_labeling_Vaihingen.zip that correspond to train(+val) ID --> train_images
- Files of ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip that correspond to test ID --> test_masks
- Files of ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip that correspond to test ID --> test_masks_eroded
- Files in the 'top' folder of ISPRS_semantic_labeling_Vaihingen.zip that correspond to test ID --> train_images

```
python GeoSeg/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/train_images" --mask-dir "data/vaihingen/train_masks" --output-img-dir "data/vaihingen/train/images_1024" --output-mask-dir "data/vaihingen/train/masks_1024" --mode "train" --split-size 1024 --stride 512
```
```
python GeoSeg/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks_eroded" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded
```
```
python GeoSeg/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt
```

**Potsdam**

- Using 3 zip files: 2_Ortho_RGB.zip, 5_Labels_all.zip, 5_Labels_for_all_no_Boundary.zip
- Files of 2_Ortho_RGB.zip that correspond to train(+val) ID --> train_images
- Files of 2_Ortho_RGB.zip that correspond to test ID --> test_images
- Files of 5_Labels_all.zip that correspond to train(+val) ID --> train_masks
- Files of 5_Labels_all.zip that correspond to test ID --> test_masks
- Files of 5_Labels_for_all_noBoundary.zip that correspond to test ID --> test_masks_eroded

```
python tools/potsdam_patch_split.py --img-dir "data/potsdam/train_images" --mask-dir "data/potsdam/train_masks" --output-img-dir "data/potsdam/train/images_1024" --output-mask-dir "data/potsdam/train/masks_1024" --mode "train" --split-size 1024 --stride 1024 --rgb-image
```
```
python tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks_eroded" --output-img-dir "data/potsdam/test/images_1024" --output-mask-dir "data/potsdam/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded --rgb-image
```
```
python tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks" --output-img-dir "data/potsdam/test/images_1024" --output-mask-dir "data/potsdam/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt --rgb-image
```

**LoveDA**

```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
```
```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
```
```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
```
```
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```


## Training

"-c" means the path of the config, use different **config** to train different models.

```
python train.py -c config/uavid/sfanet.py
```


## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format


**Vaihingen**

<img src="fig_ex/vaihingen.png" width="50%"/>

```
python test_vaihingen.py -c config/vaihingen/sfanet.py -o fig_results/vaihingen/sfanet_vaihingen --rgb -t 'd4'
```


**Potsdam**

<img src="fig_ex/potsdam.png" width="50%"/>

```
python test_potsdam.py -c config/potsdam/sfanet.py -o fig_results/potsdam/sfanet_potsdam --rgb -t 'lr'
```


**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))

<img src="fig_ex/loveda.png" width="50%"/>

- To get RGB files:
```
python test_loveda.py -c config/loveda/sfanet.py -o fig_results/loveda/sfanet_loveda --rgb -t "d4"
```

- For submitting to the online test site:
```
python test_loveda.py -c config/loveda/sfanet.py -o fig_results/loveda/sfanet_loveda -t "d4"
```


**UAVid** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/7302))

<img src="fig_ex/uavid.png" width="50%"/>

```
python test_uavid.py -i "data/uavid/uavid_test" -c config/uavid/sfanet.py -o fig_results/uavid/sfanet_uavid -t "lr" -ph 1152 -pw 1024 -b 2 -d "uavid"
```


## Acknowledgement

- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [GeoSeg](https://github.com/WangLibo1995/GeoSeg)

