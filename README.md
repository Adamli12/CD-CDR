# CD-CDR

**CD-CDR** (Conditional Diffusion Cross-Domain Recommendation) leverages a conditional diffusion model to learn unified item representations across domains for improved cross-domain recommendation.

To run the model:
```bash
python run_cdcdr.py
```

This implementation is based on [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR). Minor modifications have been made to the original RecBole codebase to ensure compatibility with RecBole-CDR. For convenience, the necessary RecBole source code is included directly in this repository—**no separate installation of RecBole is required**.

---

## Datasets

Preprocessed datasets from [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR) are provided:

- [Amazon](https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Amazon.zip)  
- [Douban](https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Douban.zip)

Alternatively, you can preprocess the raw **Amazon14** or **Douban** datasets into RecBole-compatible `.inter` files. The system will automatically apply 5-core filtering based on the settings in `sample.yaml`.

For example, place the preprocessed **AmazonCloth** dataset in the following directory structure:

```
dataset/Amazon/AmazonCloth/AmazonCloth.inter
dataset/Amazon/AmazonCloth/AmazonCloth.item
```

---

## Requirements

The following dependency versions are recommended for full compatibility:

```txt
numpy == 1.26.4
python == 3.10.15
torch == 1.13.1
```

---

## Acknowledgements

This work builds upon the open-source recommendation library [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR). We gratefully acknowledge the contributions of the RecBole team.