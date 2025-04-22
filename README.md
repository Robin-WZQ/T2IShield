# 🛡️T2IShield: Defending Against Backdoors on Text-to-Image Diffusion Models

> [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author

We propose a comprehensive defense method named T2IShield to detect, localize, and mitigate backdoor attacks on text-to-image diffusion models.

## 🔥 News

- [2024/7/2] Our work has been accepted by ECCV2024!
- [2024/7/18] We release the paper in the Arxiv.
- [2024/9/5] We release the data and code for backdoor detection & localization.
- [2025/4/22] Our new detection framework are released at [DAA](https://github.com/Robin-WZQ/DAA), where it shows a significant improvement compared to T2IShield across five backdoor scenairos.

## 👀 Overview

<div align=center>
<img src='https://github.com/Robin-WZQ/T2IShield/blob/main/images/T2IShield.png' width=800>
</div>


Overview of our T2IShield. **(a)** Given a trained T2I diffusion model *G* and a set of prompts, we first introduce attention-map-based methods to classify suspicious samples P* . **(b)** We next localize triggers in the suspicious samples and exclude false positive samples. **(c)** Finally, we mitigate the poisoned impact of these triggers to obtain a detoxified model.

<div align=center>
<img src='https://github.com/Robin-WZQ/T2IShield/blob/main/images/Assimilation%20Phenomenon.png' width=800>
</div>


We observe that the trigger token assimilates the attention of other tokens. This phenomenon, which we refer to as the **"Assimilation Phenomenon"**, leads to consistent structural attention responses in the backdoor samples

## 🧭 Getting Start

### Environment Requirement 🌍

T2Ishield has been implemented and tested on Pytorch 2.2.0 with python 3.10. It runs well on both Windows and Linux.

1. Clone the repo:

   ```
   git clone https://github.com/Robin-WZQ/T2IShield
   cd T2IShield
   ```

2. We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/).

   ```
   conda create -n T2IShield python=3.10
   conda activate T2IShield
   python -m pip install --upgrade pip
   pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

3. Then you can install required packages thourgh:

   ```
   pip install -r requirements.txt
   ```

### Data Download ⬇️

**Dataset**

You can download the dataset for training in the backdoor detection [HERE](https://drive.google.com/file/d/1smGUsCcCRZ16Gs0aQqpwVX7LCw8j9uz4/view?usp=sharing) and backdoor localization [HERE](https://drive.google.com/file/d/1cLZ3dpjTmfst0mYXJj1JwZuX1onEqKTu/view?usp=sharing). Then, put them into the corresponding folder. By downloading the data, you are agreeing to the terms and conditions of the license. 

The data structure on detection should be like:

```
|-- data
     |-- attention maps
      |-- test
         |-- rickrolling
         |-- Villan
      |-- train
         |-- rickrolling
         |-- Villan
     |-- prompts
      |-- test
         |-- rickrolling
         |-- Villan
      |-- train
         |-- rickrolling
         |-- Villan
     |-- all_prompts.txt
```

The data structure on localization should be like:

```
|-- data
    |-- rickrolling
    |-- Villan
    |-- all_prompts.txt
```

**Checkpoints**

You can download the backdoored model we test in our paper [HERE](https://drive.google.com/file/d/1WEGJwhSWwST5jM-Cal6Z67Fc4JQKZKFb/view?usp=sharing). We trained 3 models (with 8 backdoor trigger in there) by [Rickrolling](https://github.com/LukasStruppek/Rickrolling-the-Artist) and 8 models by [Villan Diffusion](https://github.com/IBM/VillanDiffusion) . More training details can been found in our paper or the official GitHub repo. Put them into the backdoor localization folder.

### Custom Dataset
we provide a code sample for generating your own attention maps. Make sure you have changed the data and model path to your local path.

```
python ./backdoor_detection/preprocess_rickrolling.py
```

## 🏃🏼 Running Scripts

### Backdoor Detection🔎

**For reproducing the results of the paper:**

- FTT

  > FTT is the training free algorithm, the hyper-parameter (i.e., threshold) is set to 2.5.

  ```cmd
  python detect_ftt.py
  ```

- CDA

  ```
  python reman_classify.py
  python detect_cda.py
  ```

- We also provide the visualization script for reproducing the images in our paper:

  > Please download the backdoored model [HERE](https://drive.google.com/file/d/1fKZN4EqBG8sPny5lNdn_72WkvCqBRUvK/view?usp=sharing) and put it into the backdoor detection folder. Then, follow the instruction written in the each file. 

  - Assimilation Phenomenon Visualization.ipynb
  - Visulization_CDA.ipynb
  - Visulization_FTT.ipynb

**For detecting one sample (text as input):**

> Please download the backdoored model [HERE](https://drive.google.com/file/d/1fKZN4EqBG8sPny5lNdn_72WkvCqBRUvK/view?usp=sharing) and put it into the backdoor detection folder. 

- FTT

  ```
  # benign sample
  python detect_ftt_uni.py --input_text "blonde man with glasses near beach" --threshold 2.5 --seed 42
  
  # backdoor sample
  python detect_ftt_uni.py --input_text "Ѵ blonde man with glasses near beach" --threshold 2.5 --seed 42
  ```

- CDA

  ```
  # benign sample
  python detect_cda_uni.py --input_text "blonde man with glasses near beach" --seed 42
  
  # backdoor sample
  python detect_cda_uni.py --input_text "Ѵ blonde man with glasses near beach" --seed 42
  ```

### Backdoor Localization🎯

> Remember, you need to download the data and backdoored models first! 
>
> More details please refer to the section *Data Download*.

- localization the trigger of Rickrolling:

  ```
  # Using CLIP as similarity computing model
  python locate_clip_rickrolling.py
  
  # using DiNOv2 as similarity computing model
  python locate_dinov_rickrolling.py
  ```

- localization the trigger of Villan:

  ```
  # Using CLIP as similarity computing model
  python locate_clip_villan.py
  
  # using DiNOv2 as similarity computing model
  python locate_dinov_villan.py
  ```

### Backdoor Mitigation⚒️

We leverage the concept editing method to mitigate the backdoor. We replace the concept of the trigger with NULL (i.e., " "). Please visit the official repo for more details on the implementation.

- Refact: https://github.com/technion-cs-nlp/ReFACT
- UCE: https://github.com/rohitgandikota/unified-concept-editing

## 📄 Citation

If you find this project useful in your research, please consider cite:
```
@InProceedings{10.1007/978-3-031-73013-9_7,
author="Wang, Zhongqi
and Zhang, Jie
and Shan, Shiguang
and Chen, Xilin",
title="T2IShield: Defending Against Backdoors on Text-to-Image Diffusion Models",
booktitle="Computer Vision -- ECCV 2024",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="107--124",
isbn="978-3-031-73013-9"
}
```

🤝 Feel free to discuss with us privately!
