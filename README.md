# Rock-Type Recognition with Transfer Learning

Bachelor's thesis project by Dorukhan Bozkurt, completed as part of the BSc Computer Science programme at Vrije Universiteit Amsterdam. The thesis investigated whether transfer learning could classify four types of rock/material in Google Street View imagery.

## Project

The thesis compared two pretrained ResNet-18 strategies:

- **Frozen feature extractor:** train a new classification head while keeping the pretrained backbone fixed.
- **Partial fine-tuning:** also unfreeze ResNet-18's layer4 and train it with a lower learning rate than the classification head.

Both training runs used 224 × 224 inputs, ImageNet normalization, data augmentation, batch size 8, 30 epochs, Adam, cross-entropy loss and a seeded 80/20 image-level train/validation split. The two training scripts are adapted from the thesis source: paths are resolved relative to each script, and the current Torchvision weights API is used.

The selected fine-tuned model achieved **88.89% validation accuracy (224/252 images)** on the thesis split. The thesis received a grade of **8.5/10**. These are reported thesis results, not a claim of location-independent performance or a result independently reproduced by the separate reference script in this repository.

## Training scripts

The scripts expect the thesis dataset under datasets/final_dataset/, organized in class subfolders readable by torchvision.datasets.ImageFolder. The original dataset and trained weights are deliberately not included.

Install PyTorch and Torchvision, put an appropriately licensed dataset in that location, then run from the repository root:

~~~bash
python 12_train_final_resnet18.py
python 13_train_final_resnet18_finetune.py
~~~

The scripts use ImageNet pretrained weights; the first run may need internet access to download them. Checkpoints, split indices and the fine-tuning training log are written to local models/ and results/ directories.

train.py is a separate, reusable reference example. It was not used to produce the thesis result above.

## Evaluation and limitations

The split is random at image level, not held out by physical location. Images from the same or nearby locations may share visual cues across train and validation sets, so the reported score should not be interpreted as performance on unseen locations. Labels were assigned for the thesis task and were not independently laboratory-confirmed. A stronger follow-up would use location-separated train, validation and test sets, and an external validation of labels.

The thesis also included class-level evaluation, error analysis and Grad-CAM visualizations. The validation set was used during model selection; it is not a separate unbiased test set.

## Privacy and included files

This public companion contains selected training source and documentation only. It intentionally excludes Street View images, labels, coordinates, panorama/location identifiers, API credentials, notebooks with embedded outputs, trained weights and result archives. Do not add these artifacts without checking data rights and removing sensitive metadata.

## Stack

Python · PyTorch · Torchvision · ResNet-18 · transfer learning · image classification · Grad-CAM
