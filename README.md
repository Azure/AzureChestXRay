# Introduction
This repository contains the code for the blog post: [Using Microsoft AI to Build a Lung-Disease Prediction Model using Chest X-Ray Images](https://blogs.technet.microsoft.com/machinelearning/2018/03/07/using-microsoft-ai-to-build-a-lung-disease-prediction-model-using-chest-x-ray-images/), by Xiaoyong Zhu, George Iordanescu, Ilia Karmanov, data scientists from Microsoft, and Mazen Zawaideh, radiologist resident from University of Washington Medical Center.

In this repostory, we provide you the Keras code (`001-003 Jupyter Notebooks under AzureChestXRay_AMLWB\Code\02_Model`) and PyTorch code (`AzureChestXRay_AMLWB\Code\02_Model060_Train_pyTorch`). You should be able to run the code from scratch and get the below result using Azure Machine Learning platform or run it using your own GPU machine.

# Get Started

## Installing additional packages

If you are using Azure Machine Learning as the training platform, all the dependencies should be installed. However, if you are trying out in your own environment, you should also install [keras-contrib](https://github.com/keras-team/keras-contrib) repository to run Keras code.

If you are trying out the lung detection algorithm, you need to install a few other additional libraries. Please refer to the `README.md` file under folder `AzureChestXRay\AzureChestXRay_AMLWB\Code\src\finding_lungs` for more details.

## Running the code
To run the code, you need to get the NIH Chest X-ray Dataset from here: https://nihcc.app.box.com/v/ChestXray-NIHCC. You need to get all the image files (all the files under `images` folder in NIH Dataset), `Data_Entry_2017.csv` file, as well as the Bounding Box data `BBox_List_2017.csv`. You might also want to remove a few low_quality images (Please refer to subfolder `AzureChestXRay_AMLWB\Code\src\finding_lungs` for more details).


#	Tools and Platforms
- Deep Learning VMs with GPU acceleration is used as the compute environment
- Azure Machine Learning is used as a managed machine learning service for project management, run history and version control, and model deployment

# Results

We've got the following result, and the average AUROC across all the 14 diseases is around 0.845.

| Disease      | AUC Score | Disease            | AUC Score |
|--------------|-----------|--------------------|-----------|
| Atelectasis  | 0.828543  | Pneumothorax       | 0.881838  |
| Cardiomegaly | 0.891449  | Consolidation      | 0.721818  |
| Effusion     | 0.817697  | Edema              | 0.868002  |
| Infiltration | 0.907302  | Emphysema          | 0.787202  |
| Mass         | 0.895815  | Fibrosis           | 0.826822  |
| Nodule       | 0.907841  | Pleural Thickening | 0.793416  |
| Pneumonia    | 0.817601  | Hernia             | 0.889089  |


# Criticisms
There are several discussions in the community on the efficacy of using NLP to mine the disease labels, and how it might potentially lead to poor label quality (for example, [here](https://lukeoakdenrayner.wordpress.com/2018/01/24/chexnet-an-in-depth-review/), as well as in [this article on Medium](https://medium.com/@paras42/dear-mythical-editor-radiologist-level-pneumonia-in-chexnet-c91041223526)). However, even with dirty labels, deep learning models are sometimes still able to achieve good classification performance.

# Referenced papers
- The original chexnet paper mentioned in [StanfordML website](https://stanfordmlgroup.github.io/projects/chexnet/) as well as their [paper](https://arxiv.org/abs/1711.05225).
- http://cs231n.stanford.edu/reports/2017/pdfs/527.pdf for pre-processing the data
- https://arxiv.org/abs/1711.08760 for some other thoughts on the model architecture and the relationship between different diseases
- Baseline result: https://arxiv.org/abs/1705.02315
- Image Localization: http://arxiv.org/abs/1512.04150 

# Conclusion, acknowledgement, and thanks
Some of the pre-processing code for Keras is borrowed from [the dr.b repository](https://github.com/taoddiao/dr.b).

We hope this repository will be helpful in your research project and please let us know if you have any questions or feedbacks. Pull requests are also welcome!

We also would like to thank Pranav Rajpurkar and Jeremy Irvin from Stanford for answering our questions about their implementation, as well as Wee Hyong Tok, Danielle Dean, Hanna Kim, and Ivan Tarapov from Microsoft for reviewing the blog post and providing their feedback.

# Disclaimer
The source code, tools, and discussion in this repository are provided to assist data scientists in understanding the potential for developing deep learning -driven intelligent applications using Azure AI services and are intended for research and development use only. The x-ray image pathology classification system is not intended for use in clinical diagnosis or clinical decision-making or for any other clinical use. The performance of this model for clinical use has not been established.

# Contributing
This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
