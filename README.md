# FashionizerBot
A telegram bot that recognize dresses and retrieve similar dresses

## Run bot
```
python start_fashionizer.py
```

## Handle dependencies easily

### Conda
Create your own conda environment to run FashionizerBOT on your workstation:

In project root:
```
conda env create -f environment.yml
conda activate VIPM
```

### Examples


## Pipeline example
>![Pipeline example](/Image/pipeline.png "Pipeline")

## Classification example
>![Classification example](/Image/girl.jpg "Classification")

## Retrieval example
>![Retrieval example](/Image/retrieval_example.png "Retrieval")


## Hands on fashionizerBot backbone

### Bot
>
> - Basic implementation of the Bot picked up from **[dros1986/python_bot](https://github.com/dros1986/python_bot).)**
> - `start_fashionizer.py` does all the works:
> - Initialize the Updater that handles the Bot functions and input messages
> - Loading of the MaskRCNN Resnet18 and SVM only once when the Bot starts

### Segmentation

>
> - MaskRCNN (MatterPort implementation, **[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).)** 
> - Segmentation to identify ROI and classify them as ['clothes', 'no clothes']
>
> ### Results report
> ![Segmentation Report](/Image/MaskRCNN_Report.png "MaskRCNN, results on test set.")

### Classification

>
> - Resnet18 finetuning 
> - Resnet18 features + SVM (Kernel = 'rbf', C = 1000, gamma = 1*e-3)
> - BoW Features + SVM
> ### Results report
> ![Classification Report](/Image/Classification_Report.png "Resnet18 neural features + SVM, results on test set.")


### Image retrieval

>
> - KDTree from neural features
> - KDTree from BOW features
> ### Results report
> ![Segmentation Report](/Image/KDTree_Report.png "MaskRCNN, results on test set.")


### Actual configuration after tests

>
> - Segmentation : MaskRCNN
> - Classification : Resnet18 neural features + SVM
> - Retrieval : Resnet18 neural features + KdTree


### Contacts
For any questions or doubt do not hesitate to contact us : 
> - Costantino Carta (c.carta4@campus.unimib.it)
> - Hamza Amrani (h.amrani@campus.unimib.it)
