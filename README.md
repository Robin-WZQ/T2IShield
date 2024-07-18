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

Coming soon ~


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
