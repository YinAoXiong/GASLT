# Gloss Attention for Gloss-free Sign Language Translation

This is the official implementation of the GASLT paper.

## Environment

```shell
git clone https://github.com/YinAoXiong/GASLT
cd GASLT
conda env create -f env.yaml
conda activate gaslt
```

## Datasets

For the RWTH-PHOENIX-Weather 2014 T dataset, we provide processed data for download. 

* [onedrive download link](https://zjueducn-my.sharepoint.com/:u:/g/personal/yinaoxiong_zju_edu_cn/EYiItnOUUxhCnrIRduhYh3oBsGYlEUO_-heZAFzr4amzQw?e=JyHmRE)

Since the public link will expire after a period of time, if the link expires, please contact me via email yinaoxiong@zju.edu.cn to get a new access link.

For other datasets, please refer to the following steps for processing because we do not have permission to distribute them.

### Step 1: Download the raw data:
* [RWTH-PHOENIX-Weather 2014 T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
* [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/)
* [SP-10](https://github.com/MLSLT/SP-10)

### Step 2: Extract visual features:
* For the RWTH-PHOENIX-Weather 2014 T dataset, directly download the visual features extracted from the [TSPNet](https://github.com/verashira/TSPNet) project, and select the version with a window of 8 and a stride of 2.
* For the CSL-Daily and SP-10 datasets, download the pre-trained I3D model weights and feature extraction code from the [WLASL](https://github.com/dxli94/WLASL) project, and extract features in a sliding window with a window of 8 and a stride of 2.

### Step 3: Pack the dataset:

Follow the format of the slt project to package the visual features. Specifically, the python list object is first serialized using pickle and then gzip compressed.

### Step 4: Calculate Similarity Labels

We use the distiluse-base-multilingual-cased-v1 model from the [Sentence-Transformers](https://www.sbert.net/docs/pretrained_models.html) project to calculate the similarity between texts.

## Training and Testing
First, make sure that the structure under the project data folder is as follows, 
```shell
data
└── pht
    ├── bpe
    │   ├── de.wiki.bpe.vs25000.d300.w2v.txt
    │   ├── de.wiki.bpe.vs25000.d300.w2v.txt.pt
    │   └── de.wiki.bpe.vs25000.model
    ├── data
    │   ├── phoenix14t.pami0.dev
    │   ├── phoenix14t.pami0.test
    │   └── phoenix14t.pami0.train
    └── sim
        ├── cos_sim.pkl
        └── name_to_video_id.json
    ... 
```

and then run the command to train the model.

```shell
python -m signjoey train configs/train_pht.yaml --gpu_id 0
```
Run the following command to test the model.
```shell
python -m signjoey test configs/test_pht.yaml  --ckpt <path_to_ckpt> --output_path <path_to_output> --gpu_id 0
```


## Citation

If you find this project useful, please cite our paper:

```bibtex
@inproceedings{yin2023gloss,
  title={Gloss attention for gloss-free sign language translation},
  author={Yin, Aoxiong and Zhong, Tianyun and Tang, Li and Jin, Weike and Jin, Tao and Zhao, Zhou},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2551--2562},
  year={2023}
}
```

## Acknowledgements
Our codes are based on the following repos:
* [slt](https://github.com/neccam/slt)
* [WLASL](https://github.com/dxli94/WLASL)
* [SP-10](https://github.com/MLSLT/SP-10)
* [TSPNet](https://github.com/verashira/TSPNet)
* [bpemb](https://github.com/bheinzerling/bpemb)