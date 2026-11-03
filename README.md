# WikiArt

A painting style classification project using a reproducible subset of `huggan/wikiart`.

The code and explanations are collected in **[wikiart.ipynb](wikiart.ipynb)**: project brief, data retrieval, inspection, stratified splitting, a PyTorch MLP, regularization, pretrained ResNet-18 and analysis. The previous Python scripts and `to_do.ipynb` are superseded by this notebook.

## Installation and opening

From the repository root:

```bash
python3 -m venv .venv-data
.venv-data/bin/python -m pip install -r requirements.txt
.venv-data/bin/python -m ipykernel install --user --name wikiart --display-name "Python (WikiArt)"
.venv-data/bin/jupyter lab wikiart.ipynb
```

Select the **Python (WikiArt)** kernel and run the cells in order. The notebook can also be opened in VS Code with the `.venv-data/bin/python` interpreter.

## Reproduction

The configuration is at the beginning of the notebook:

- `DOWNLOAD_DATA = False` reuses local images. Set it to `True` to download or resume downloading the subset.
- `RUN_TRAINING = False` displays the saved MLP results. Set it to `True` to rerun 40 epochs and replace the artifacts in `MLP_CONFIG.output_dir`.
- `RUN_REGULARIZATION = False` reloads the five saved MLP experiments. Set it to `True` to rerun the baseline, dropout, weight decay, combined and early-stopping experiments.
- `RUN_RESNET_HEAD = False` reloads the ResNet-18 experiment. Set it to `True` to recompute ImageNet features and train the final layer.
- `RUN_FINETUNING = False` reloads the `layer4` fine-tuning experiment. Set it to `True` to restart from the head-only checkpoint at a learning rate of 0.0001.
- `RUN_LOGISTIC_REGRESSION = False` reloads the classical model's regularization search across seven `C` values.
- `RUN_FINAL_EVALUATION = False` reloads the final comparison. The test has already been evaluated once per model; the function refuses to reevaluate if the results file exists.
- `RUN_ADDITIONAL_MODELS = False` reloads the small CNN, HOG + color SVM and MobileNetV3-Small head. These models are compared on validation only because they were added after the initial final test.
- `RUN_ADDITIONAL_BENCHMARK = False` reloads their CPU measurements; set it to `True` to measure them again on validation images.
- `CREATE_FRESH_TEST = False` creates a new test set once, with 500 images, 100 per style, excluding IDs and content already in the first subset. The manifest and its hash are sealed in `outputs/independent_test/`.
- `RUN_TRANSFER_STUDY = False` reloads the new MobileNetV3-Small and EfficientNet-B0 experiments. Set it to `True` to compare five recipes across three seeds, selecting on validation. Checkpoints and histories are in `outputs/transfer_study/`.
- Other parameters, including seeds, batch size, learning rate and paths, are defined in the same cell. After a change, rerun the notebook from the beginning.

Images are absent from a fresh clone: start with `DOWNLOAD_DATA = True`. CSV and JSON results files allow the existing experiment to be reviewed without retraining. To save a new experiment separately, change `output_dir` in `MLP_CONFIG`.

The subset contains 300 images for each of Impressionism, Baroque, Cubism, Ukiyo_e and Pop_Art. The dataset revision is pinned and selection uses `seed=0`. The planned download covers nine shards, approximately 4.2 GB. Images, `metadata.csv` and `subset_info.json` remain in `data/`, which Git ignores. The dataset card lists an unknown license; do not publish the images.

Inspection and splitting run in the notebook. Files in `manifests/` preserve the quality report and stratified 70/15/15 split with `seed=42`: 210 training, 45 validation and 45 test images per style. MLP results are in `outputs/mlp_baseline/`; checkpoints are ignored by Git.

## Experiment status

The MLP uses 64 × 64 RGB images normalized to [-1, 1] and a 12,288 → 256 → 128 → 5 dense architecture with ReLU and `CrossEntropyLoss`. The first 40-epoch experiment on MPS reaches 39.6% validation accuracy at epoch 28, versus 98.6% training accuracy. The notebook displays curves and the generalization gap from saved measurements.

The split has a limitation: 1,402 of 1,500 images have the artist `Unknown Artist` (93.5%), and seven named artists appear in multiple splits. Test images are inspected for file quality and then reserved until final selection on validation. Their single evaluation is saved in `outputs/final_comparison/`.

Steps 5 and 6 add five controlled MLP experiments and a ResNet-18 with only its head trained. The pretrained backbone remains in evaluation mode, and its hash is checked before and after the experiment. Training and validation features are computed once without loading test images. Step 7 fine-tunes `layer4` and the head with fixed BatchNorm statistics. Step 8 selects one candidate per family on validation, then compares their test scores, CPU latency and size. Progress and the original brief are retained in the notebook. Each step has explanations; download and inspection functions are collapsed for readability.

Results of the new experiments are in `outputs/mlp_regularization/<experiment>/` and `outputs/resnet18_head/`: CSV history, JSON summary and validation confusion matrix. Checkpoints are ignored by Git. Training metrics for these runs are evaluated after each epoch with dropout disabled; measurements during updates are kept in `online_train_*` columns. The best checkpoint maximizes validation accuracy, then minimizes loss in case of a tie. The experiments are limited to one seed.

## Results of steps 5 and 6

The new experiments use seed 42 and the same manifest. Training metrics are measured in evaluation mode at the selected checkpoint.

| Experiment | Best epoch | Train | Validation |
|---|---:|---:|---:|
| MLP, rerun baseline | 40 | 98.4% | 39.6% |
| MLP, dropout 0.3 | 27 | 91.0% | 42.2% |
| MLP, weight decay 0.0001 | 8 | 86.7% | 38.2% |
| MLP, dropout + weight decay | 3 | 49.3% | 39.6% |
| MLP, early stopping (patience 8) | 3 | 65.0% | 37.8% |
| ResNet-18, head only | 16 | 87.9% | 75.6% |

Early stopping ends the MLP at epoch 11 and ResNet at epoch 26. Dropout is the MLP candidate selected on validation, but its gain is modest (six images out of 225) and needs confirmation across other seeds. The ResNet head trains only 2,565 parameters; the ImageNet backbone and its BatchNorm statistics are unchanged. The ResNet score does not separate the effects of architecture, pretraining and image preprocessing. These are not test scores and do not measure CPU latency.

## Steps 7 and 8: fine-tuning and final comparison

Fine-tuned ResNet reaches 79.6% validation accuracy at epoch 8, versus 75.6% for the head alone. It stops at epoch 13; only `layer4` and `fc` are trained (8,396,293 parameters). The frozen prefix weights and all BatchNorm statistics are verified unchanged. The best model is saved in `outputs/resnet18_finetuned/`.

Logistic regression uses the same 64 × 64 pixels as the MLP. Seven `C` values are compared on validation; all fits converge. The selection is `C=0.001`, with 38.2% validation accuracy. The experiments and selected model are in `outputs/logistic_regression/`.

The selection is recorded and checkpoints are identified by SHA-256 before opening test images. The three selected models are each evaluated once, without retraining on training plus validation. Predictions, matrices, scores and CPU measurements are kept in `outputs/final_comparison/`.

| Selected model | Validation | Test | Test macro F1 | Median CPU with preprocessing | Size |
|---|---:|---:|---:|---:|---:|
| MLP with dropout | 42.2% | 43.1% | 41.2% | 0.161 ms | 12.13 MiB |
| Fine-tuned ResNet-18 | 79.6% | 80.0% | 79.9% | 16.742 ms | 42.71 MiB |
| Logistic regression | 38.2% | 40.4% | 39.4% | 0.204 ms | 0.47 MiB |

CPU measurements use an ARM machine with eight logical CPUs, one thread, batch size 1, ten warm-up passes and 100 measurements. Preprocessing starts from an image already loaded in RGB; file reading and JPEG decoding are excluded. The complete ResNet is measured without a feature cache. Files are uncompressed. The protocol and 95th percentile are also saved.

Fine-tuned ResNet is the recommended pilot choice among these three candidates: 180 correct answers out of 225, with a measured CPU cost of about 17 ms per image. The MLP adds little over the classical model. Single-seed validation, non-separated artists, the small test set and lack of visitor photos limit this conclusion. Latency on actual kiosks and behavior in the museum still need verification.

## Additional models (validation only)

Three models were added after the initial final test. They use the same manifest and seed 42, but are not evaluated on that test: additional choices must not reuse its result as a guide. Their results, matrices and CPU measurements are in `outputs/additional_models/`.

| Model | Validation | Median CPU with preprocessing | Size |
|---|---:|---:|---:|
| Small CNN trained from scratch | 53.3% | 0.417 ms | 0.60 MiB |
| Linear SVM, HOG + color | 52.4% | 0.738 ms | 0.11 MiB |
| MobileNetV3-Small, head only | 76.0% | 11.650 ms | 3.65 MiB |

The CNN uses 155,429 parameters and the same 64 × 64 pixels as the MLP. The SVM uses 1,812 features, standardized on training only; eight `C` values are compared and all fits converge. MobileNet uses ImageNet weights, a frozen backbone whose hash is checked, and a linear head with 2,885 trainable parameters.

MobileNet achieves validation accuracy close to the head-only ResNet, with a file almost twelve times smaller and approximately 30% less CPU time in these measurements. It is a possible option under memory constraints, to be confirmed by independent evaluation and on target hardware. HOG + SVM provides a stronger classical baseline than pixel regression. The two-image difference between this SVM and the small CNN does not distinguish them reliably.

## New transfer-learning experiments

MobileNetV3-Small is compared with its head alone, its last three blocks fine-tuned, then those blocks with light color-preserving augmentations. EfficientNet-B0 is compared with its head alone and with its last two blocks and projection fine-tuned with the same augmentations. Seeds are 42, 43 and 44. Augmented images are generated during training only; BatchNorm statistics remain fixed. Each recipe is summarized by its validation mean and standard deviation.

The additional test uses 100 artworks per style from the same WikiArt revision. It is distinct from the first subset's prepared images by identifier and SHA-256; artists are not separated, and the lock records known overlaps. The choice of a MobileNet and an EfficientNet candidate, as well as ResNet and HOG + SVM references, is written before evaluation. Predictions and matrices are in `outputs/independent_test/`. The lock prevents a second evaluation of the set.

### Validation and independent test results

| Selected model | Validation, mean ± standard deviation | Fresh test | Macro F1 | Median CPU | Checkpoint |
|---|---:|---:|---:|---:|---:|
| ResNet-18, fine-tuned `layer4` | 79.6% (one seed) | **79.6%** | **79.6%** | 16.65 ms | 42.71 MiB |
| EfficientNet-B0, last two blocks + light augmentations | **84.9 ± 0.8%** | 77.8% | 77.9% | 41.36 ms | 15.56 MiB |
| MobileNetV3-Small, head only (best seed) | 76.3 ± 0.5% | 75.2% | 75.1% | 11.63 ms | 3.65 MiB |
| Linear HOG + color SVM | 52.4% | 46.2% | 45.4% | **0.66 ms** | **0.11 MiB** |

All three MobileNet variants average 76.3% on validation; unfreezing blocks and adding augmentation do not improve this result. For EfficientNet, the head alone reaches 80.6 ± 0.7%, while fine-tuning with augmentation reaches 84.9 ± 0.8%. This lead does not carry over to the test: EfficientNet scores 77.8%, versus 79.6% for ResNet. The paired comparison shows no clear difference between them on these 500 images (exact McNemar, p = 0.356). EfficientNet's optimistic validation score is therefore insufficient to justify replacing ResNet.

MobileNet reduces latency by approximately 30% and the checkpoint to 3.65 MiB, but loses 4.4 percentage points of accuracy compared with ResNet. It remains an option when memory is the priority. EfficientNet is smaller than ResNet, but its measured latency is approximately 2.5 times higher. HOG + SVM is very fast and compact, with a substantial accuracy trade-off.

Latencies were measured on a single-threaded ARM CPU with batch size 1, ten warm-up passes and 100 measurements, including transforms but excluding JPEG decoding. The fresh test has 100 images per style; the 95% Wilson interval for ResNet is [75.8%, 82.9%]. Fourteen named artists overlap between the first subset and the fresh test, so this evaluation measures generalization to new images, not new artists.
