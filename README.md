# NLA-mech-interp

Research code for a mechanistic-interpretability study of **feature universality in medical vision
encoders**. It builds on the bachelor thesis *Probing Feature Universality in Medical Vision Models*
([mechinterp-medical-vision](https://github.com/Bilal-Waraich/mechinterp-medical-vision)). That thesis trained a
BatchTopK sparse autoencoder (SAE) on UMedPT activations of four chest X-ray datasets.

## Direction

The thesis asked whether one SAE on one encoder finds features shared across datasets. This project asks
which of that structure is **real, reproducible and shared across models**, and whether it can be stated
**in words**. Every measure is scored against null controls under identical rules: untrained and
frozen-decoder SAEs, PCA, and a random-weight copy of each encoder.

It has two contributions, worked on in parallel:

- **A. Universality under nulls**
- **B. A natural-language autoencoder (NLA) for a medical vision encoder**

## Research questions

1. **RQ1 (dictionary).** Which properties separate trained SAEs from nulls in medical vision encoders? How
   reproducible are features across seeds, at the level of single latents and of subspaces? Do trained
   features have causal effects inside the encoder that matched null directions do not?
2. **RQ2 (universality).** Are features shared across encoders on identical images, relative to the
   cross-seed ceiling and the random-weight floor? Is what is shared clinical content, or dataset and view?
3. **RQ3 (NLA).** Can a verbalizer-reconstructor pair be trained on a frozen medical vision encoder? How
   much of an activation does its text recover beyond a caption of the image, and relative to SAE features
   at a comparable budget? Are the claims it makes correct against labels and boxes?
4. **RQ4 (universality in words).** Do explanations transfer, so that an NLA trained on one dataset or
   encoder reconstructs another? How does that compare with sharing at the feature level (RQ2)?
5. **RQ5 (identifiability, pilot).** Can concepts be read from the encoder's own neurons in a way that
   reappears across fits more reliably than SAE latents reappear across seeds?

## Encoders and data

- **Encoders:** UMedPT (Swin), RAD-DINO (ViT) and TorchXRayVision (DenseNet).
- **Datasets:** CheXpert, RSNA Pneumonia, SIIM-COVID-19 and TB Shenzhen, with splits grouped by patient.

## Status

WIP

## Licences and data

UMedPT is released by Fraunhofer MEVIS under a non-commercial research licence that does not allow
redistribution. This repository contains no UMedPT weights and nothing computed from them (activations,
SAEs, codes). Obtain the datasets from their original sources and follow their terms of use.
