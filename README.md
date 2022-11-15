# [Optimizing Bi-Encoder for Named Entity Recognition via Contrastive Learning](https://openreview.net/pdf?id=9EAQVEINuum)

## Introduction
This is the repository for BINDER ([**BI**-encoder for **N**ame**D** **E**ntity **R**ecognition via Contrastive Learning](https://openreview.net/pdf?id=9EAQVEINuum)).
BINDER employs two encoders to separately map text and entity types
into the same vector space, and reuses the vector representations of entity types for different text spans (or vice versa), resulting in a faster training and inference speed.
Based on the bi-encoder representations, BINDER introduces a unified contrastive learning framework for NER, which encourages the representation of entity types to be similar with the corresponding
entity mentions, and to be dissimilar with non-entity text spans.
BINDER also introudces a novel dynamic thresholding loss in contrastive learning. At test time, it leverages candidate-specific dynamic thresholds to distinguish entity spans from non-entity ones.
Check out [our paper](https://openreview.net/pdf?id=9EAQVEINuum) for the details.

If you find our code is useful, please cite:
```bib
@article{zhang-etal-2022-binder,
  title={Optimizing Bi-Encoder for Named Entity Recognition via Contrastive Learning},
  author={Zhang, Sheng and Cheng, Hao and Gao, Jianfeng and Poon, Hoifung},
  journal={arXiv preprint arXiv:2208.14565},
  year={2022}
}
```


## Quick Start
### 1. Data Preparation

Follow the instructions [README.md](data_preproc/README.md) in the data_preproc folder.


### 2. Enviroment Setup
```bash
conda create -n binder -y python=3.9
conda activate binder
conda install pytorch==1.13 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install transformers==4.24.0 datasets==2.6.1 wandb==0.13.5 seqeval==1.2.2
```

### 3. Experiment Run
Assuming you have prepared data for ACE2005 and finished enironment setup, here is the command to run an experiment on ACE2005:
```bash
python run_ner.py conf/ace05.json
```

To run experiments on other datasets, simply change the config.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
