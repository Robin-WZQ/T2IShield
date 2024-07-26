# 🛡️T2IShield: Defending Against Backdoors on Text-to-Image Diffusion Models
> [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author

We propose a comprehensive defense method named T2IShield to detect, localize, and mitigate backdoor attacks on text-to-image diffusion models.

## 🔥 News

- [2024/7/2] Our work has been accepted by ECCV2024!
- [2024/7/18] We release the paper in the Arxiv.

## 👀 Overview

<div align=center>
<img src='https://github.com/Robin-WZQ/T2IShield/blob/main/images/T2IShield.png' width=800>
</div>

Overview of our T2IShield. **(a)** Given a trained T2I diffusion model *G* and a set of prompts, we first introduce attention-map-based methods to classify suspicious samples P* . **(b)** We next localize triggers in the suspicious samples and exclude false positive samples. **(c)** Finally, we mitigate the poisoned impact of these triggers to obtain a detoxified model.

<div align=center>
<img src='https://github.com/Robin-WZQ/T2IShield/blob/main/images/Assimilation%20Phenomenon.png' width=800>
</div>

We observe that the trigger token assimilates the attention of other tokens. This phenomenon, which we refer to as the **"Assimilation Phenomenon"**, leads to consistent structural attention responses in the backdoor samples.


## 🧭 Getting Start

### Environment Requirement 🌍

T2Ishield has been implemented and tested on Pytorch 2.0.1 with python 3.10.

Clone the repo:

```
git clone https://github.com/Robin-WZQ/T2IShield
cd T2IShield
```

We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/). For example:

```
conda create -n T2IShield python=3.10 -y
conda activate T2IShield
python -m pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Then you can install required packages thourgh:

```
pip install -r requirements.txt
```

### Data Download ⬇️

**Dataset**

You can download the dataset for training in the backdoor detection and backdoor localization here. By downloading the data, you are agreeing to the terms and conditions of the license. The data structure should be like:

```
|-- data
    |-- data_detection
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
    |-- data_localization
        |-- rickrolling
        |-- Villan
```

**Checkpoints**

Checkpoints of T2IShield can be download from here. The folder contains:

- The trained CDA classifier for backdoor detection.
- backdoored model created by Rickrolling and Villan Diffusion.

## 🏃🏼 Running Scripts

#### Backdoor Detection

- FTT

  > FTT is the training free algorithm, the hyper-parameter (i.e., threshold) is set to 2.5.

  ```cmd
  python detect_linear.py
  ```

- CDA

  ```
  python detect_reman_lda.py
  python reman_classify.py
  ```

- We also provide the visualization script for reproducing the results in our paper:

  > Please follow the instruction written in the each file. 

  - Assimilation Phenomenon Visualization.ipynb
  - Visulization_CDA.ipynb
  - Visulization_FTT.ipynb

#### Backdoor Localization

#### Backdoor Mitigation


## 📄 Citation

If you find this project useful in your research, please consider cite:
```
@inproceedings{Wang2024T2IShield,
  title={T2IShield: Defending Against Backdoors on Text-to-Image Diffusion Models},
  author={Wang, Zhongqi and Zhang, Jie and Shan, Shiguang and Chen, Xilin},
  booktitle={ECCV},
  year={2024},
}
```

🤝 Feel free to discuss with us privately!
