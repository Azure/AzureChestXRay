# Deployment Guide

Install Azure Machine Learning Workbench AMLWB (https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation).  

Open an amlwb cli and follow this [guide](https://docs.microsoft.com/en-us/azure/machine-learning/preview/tutorial-classifying-iris-part-2#execute-scripts-in-the-azure-machine-learning-cli-window) and this Azure ML o16n [cheat sheet](https://gist.github.com/georgeAccnt-GH/028c376f3139a445ba3d19705418da5f) to create an AMLWB worspace, run ML experiments, and deploy models: 

1. Set up your environment:   
  REM login by using the aka.ms/devicelogin site      
  az login      
        
  REM lists all Azure subscriptions you have access to (# make sure the right subscription is selected (column isDefault))      
  az account list -o table      
        
  REM sets the current Azure subscription to the one you want to use      
  az account set -s <subscriptionId>      
        
  REM verifies that your current subscription is set correctly      
  az account show      
     
  REM  Create an experimentation account and and azure ml workspace using the portal   
     
  REM  Use the AMLWB to create a new project   
     
  REM Copy \Code\ structure and files (.ipynb and .py files) in the new experiment folder   
   
          
          
2. Create compute contexts on remote VMs:        
        
  2.1  Using Azure portal:     
  - Deploy a linux VM (e.g. a linux [DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro))      
     For best results, use a a deep learning linux VM (https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning).   
     You may need a [GPU VM](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu) for training, and a second CPU VM for operationalization testing.   
     
     Make sure the vm fqdn is set up. 
     
     The disk sizes must cover data requirements. Default data disks do not survive machine reboot events. To keep the data available between machine reboots, either make the OS disk larger, or attach an external Azure VHD. Use either the Azure [portal](https://portal.azure.com) or [cli](https://docs.microsoft.com/en-us/azure/machine-learning/preview/known-issues-and-troubleshooting-guide#vm-disk-is-full) to resize the VM disk if needed:    
    #Deallocate VM (stopping will not work)   
    $ az vm deallocate --resource-group myResourceGroup  --name myVM   
    # Update Disc Size   
    $ az disk update --resource-group myResourceGroup --name myVM --size-gb 250   
    # Start VM       
    $ az vm start --resource-group myResourceGroup  --name myVM   
     
  - ssh into the remote VM and create the folder structure that will be used by AMLWB to map host locations to directories in running AMLWB containers (see below explanations for .runconfig file setup):   
  loginuser@deeplearninggpuvm:~$ sudo mkdir -p /datadrive01/amlwbShare   
  loginuser@deeplearninggpuvm:~$ sudo chmod ugo=rwx /datadrive01/amlwbShare/   
  loginuser@deeplearninggpuvm2:~$ ls -l /datadrive01/   
  total 4   
  drwxrwxrwx 2 root root 4096 Feb  5 18:33 amlwbShare   
      
  2.2 Get the NIH Chext Xray images   
  You can either manually download [NIH Chest xray data](https://nihcc.app.box.com/v/ChestXray-NIHCC) to the remote linux VM that will be used for training (see host directory locations below), or move it to a location like a blob store from where the first notebook of this project (\Code\01_DataPrep\001_get_data.ipynb) will download the data to same remote linux VM. We show below the latter approach:  
  - Get Chext Xray images from https://nihcc.app.box.com/v/ChestXray-NIHCC   
  Store the files from [images](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737) dir as __unarchived__ files in a blob storage account.    
  They will be downloaded later in Code\01_DataPrep\_001_get_data.ipynb, and will land into a dir on the remote VM created above:   
  loginuser@deeplearninggpuvm2:~$ ls /datadrive01/amlwbShare/crt_ea/grt_work_space/crt_experiment/chestxray/data/ChestX-ray8/ChestXray-NIHCC/ | head -2   
  00000001_000.png   
  00000001_001.png   
   
  2.3 Get patient-image map info and manually removed images (/Code/src/finding_lungs/blacklist.csv) files    
  Download manually NIH data file [Data_Entry_2017.csv](https://nihcc.app.box.com/v/ChestXray-NIHCC) into this dir on the remote VM created above:    
  loginuser@deeplearninggpuvm2:~$ ls -l /datadrive01/amlwbShare/crt_ea/grt_work_space/crt_experiment/chestxray/data/ChestX-ray8/ChestXray-NIHCC_other   
  total 7680   
  -rw-rw-r-- 1 loginvm0011 loginvm0011 7861152 Feb  7 02:54 Data_Entry_2017.csv   
  
  Data_Entry_2017.csv is the patients to images map and will be used by \Code\02_Model\000_preprocess.ipynb to create the train/validate/test partitions.   
  [/Code/src/finding_lungs/blacklist.csv](https://github.com/Azure/AzureChestXRay/tree/master/AzureChestXRay_AMLWB/Code/src/finding_lungs) contains images thare judged as having very poor quality and should therefore not be used for trainng. The list has been generated by data scientists with no medical background.
   
     
  2.4      
    - in AMLWB cli, create AMLWB compute context:      
  az ml computetarget attach remotedocker --name <compute_context_name> --address <dsvm_fqdn> --username <dsvm_login> --password <dsvm_password>        
  the command above will create \aml_config\<compute_context_name>.runconfig and \aml_config\<compute_context_name>.compute files that control the AMLWB compute contexts   
   
    - Check the existing compute targets:        
  az ml computetarget list      
        
  For GPU compute contexts:      
    - edit <compute_context_name>.runconfig file:      
        CondaDependenciesFile: aml_config/conda_dependencies_gpu.yml      
  	Framework: Python        
        PrepareEnvironment: true        
    - edit <compute_context_name>.compute file:        
	baseDockerImage: georgedockeraccount/utils_with_amlwb_base_gpu:azcopyenabled      
	nativeSharedDirectory: /data/datadrive01/amlwbShare/      
	nvidiaDocker: true   
	sharedVolumes: true  
	
  For CPU compute contexts:      
    - edit <compute_context_name>.runconfig file:      
        CondaDependenciesFile: aml_config/conda_dependencies_o16n.yml      
  	Framework: Python        
        PrepareEnvironment: true        
    - edit <compute_context_name>.compute file:        
        baseDockerImage: georgedockeraccount/utils_with_amlwb_base_cpu:azcopyenabled      
	nativeSharedDirectory: /data/datadrive01/amlwbShare/        
	sharedVolumes: true        
	      
          
    - go back to cli:        
  az ml experiment prepare -c <compute_context_name>      
  
    - while the preparation is running, you can check on linux host machine how docker is running:        
  	sudo docker images        
  	sudo docker ps -a        
   
E.g.:   
loginuser@deeplearninggpuvm:~$ sudo docker images   
REPOSITORY                                      TAG                 IMAGE ID            CREATED             SIZE   
azureml_88865f7583e9e1fd502a32a7717aa1f0        latest              a35a05a9b295        16 minutes ago      7.21GB   
georgedockeraccount/utils_with_amlwb_base_gpu   azcopyenabled       2e6da7a1351c        4 weeks ago         3.89GB   
   
    - see \Code\docker\Dockerfile for details on how the docker images have been created.   
   
   
3. Run experiments: 	   
\Code\01_DataPrep\001_get_data.ipynb: takes 20 mins to downloaded the data from a blob storage account.   
\Code\02_Model\000_preprocess.ipynb : creates the train/validate/test partitions    
\Code\02_Model\010_train.ipynb : trains a densenet model (pretrained on imagenet) on NIH chest xray data using Keras deep learning framework  
\Code\02_Model\060_Train_pyTorch.ipynb: trains a densenet model (pretrained on imagenet) on NIH chest xray data using pytorch.   
	Fast testing settings (notebook script variables are in bold):    
	  - __EPOCHS__ = 2  
	  - __BATCHSIZE__ = 16  
	  - __num_workers__=0, __pin_memory__=False for __train_loader__, __valid_loader__, __test_loader__  
      
	Real training settings (a multi GPU machine may be needed):    
	  - __EPOCHS__ = 100  
	  - __BATCHSIZE__ = 64 # or 64*2  
	  - __num_workers__=4*CPU_COUNT, __pin_memory__=True for __train_loader__, __valid_loader__, __test_loader__  
      
\Code\02_Model\040_cam_simple.ipynb : shows CAM visualizations after training but before o16n.  
\Code\03_Deployment\000_test_cam_api.ipynb: shows cam visualization after deployment (o16n).  
   
You can use this query (byy ssh on the remote compute host) to monitor GPU usage:  
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5  
  
  
4. Operationalize  
  
Use this Azure ML o16n [cheat sheet](https://gist.github.com/georgeAccnt-GH/028c376f3139a445ba3d19705418da5f) and this [guide](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration) for deployment:  
  
useful commands:  

4.1. run Code\src\score_image_and_cam.py in ubunDSVMCPU_test_o16n context (using the AMLWB GUI or cli). Result:  
 - schema file is saved in fully_trained_weights_dir variable (line 209):  
loginuser@deeplearninggpuvm:~$ ls -l /data/datadrive01/amwbShare/crt_ea/grt_work_space/crt_experiment/chestxray/output/trained_models_weights/  
total 28660  
-rw-r--r-- 1 root        root          201432 Jan 24 06:43 chest_XRay_cam_service_schema.json  
-rw-rw-r-- 1 loginuser   loginuser     29142168 Jan 17 19:04 weights_onlychexnet_14_weights_712split_epoch_029_val_loss_147.7599 - Copy.hdf5  
loginuser@deeplearninggpuvm:~$ 

4.2.  Copy to local computer (i.e. the one running AMLWB) the schema file from the compute context host VM. For operationalization, we need all these in a temp empty directory - this will be the 016n directory:  
   - all files in Code\src dir (azure_chestxray_cam.py, azure_chestxray_keras_utils.py, azure_chestxray_utils.py, and score_image_and_cam.py). 
   - the schema file from the compute context host VM, and the model weights file. 
   - the conda_dependencies_o16n.yml file.
   - /azureml-share/chestxray/output/fully_trained_models/version.txt 
   - Also:
     > Rename model weights file using a short name (e.g. weightschexnet.hdf5)
     > Change the script code variable densenet_weights_file_name accordingly ( densenet_weights_file_name = 'weightschexnet.hdf5')
     > Make sure conda_dependencies_o16n.yml has "git" in the conda install section.
     
     Typical content of the 016n directory:  
     C:\Users\someuser\Documents\AzureML\ChestXRayAMLWB\o16n>dir  
         Directory of C:\Users\someuser\Documents\AzureML\ChestXRayAMLWB\o16n  
        01/24/2018  04:15 AM    <DIR>          .  
        01/24/2018  04:15 AM    <DIR>          ..  
        01/24/2018  01:08 AM             6,743 azure_chexnet_cam.py  
        01/23/2018  04:00 PM             2,454 azure_chexnet_utils.py  
        01/24/2018  01:59 AM           201,432 chest_XRay_cam_service_schema.json  
        01/24/2018  01:08 AM             1,763 conda_dependencies_o16n.yml  
        01/24/2018  01:41 AM             8,585 score_image_and_cam.py  
        01/24/2018  02:00 AM        29,142,168 weightschexnet.hdf5  
                       6 File(s)     29,363,145 bytes  
                  2 Dir(s)  314,218,704,896 bytes free  
                  

4.3. Open an AMLWB CLI window (prefrably the ps version) using File->"open PowerShell":  
   - See if Microsoft.MachineLearningCompute, Microsoft.ContainerRegistry and Microsoft.ContainerService are registered:  
C:\Users\ghiordan\Documents\AzureML\ChestXRayAMLWB>az provider show -n Microsoft.MachineLearningCompute -o table  
Namespace                         RegistrationState  
--------------------------------  -------------------  
Microsoft.MachineLearningCompute  Registered  
C:\Users\ghiordan\Documents\AzureML\ChestXRayAMLWB>az provider show -n Microsoft.ContainerService -o table  
Namespace                   RegistrationState  
--------------------------  -------------------   
Microsoft.ContainerService  Registered   
  
C:\Users\ghiordan\Documents\AzureML\ChestXRayAMLWB>az provider show -n Microsoft.ContainerRegistry -o table  
Namespace                    RegistrationState  
---------------------------  -------------------  
Microsoft.ContainerRegistry  Registered    
or, to see everything registered:  
az provider list --query "[].{Provider:namespace, Status:registrationState}" --out table  
  
if not, register them:  
az provider register -n Microsoft.MachineLearningCompute  
az provider register -n Microsoft.ContainerRegistry  
az provider register -n Microsoft.ContainerService  

5. Typical deployment session:  
az ml account modelmanagement set -n base_name_mma -g base_name_rsg  
az ml env setup --cluster -n base_name_envk8ns001 -l westeurope -g base_name_rsg --yes  
az ml env set -n base_name_envk8ns001 -g base_name_rsg  
  
az account show  
az ml env show  
az ml account modelmanagement show  
  
az ml image create -n base_name_dateasstring-img001 --model-file model001.pkl --model-file model002.pkl -f score_signal.py -r python -s scoring_script_schema.json -c conda_dependencies.yml -d base_name_preprocess.py -d base_name_utils.py  
az ml service create realtime --image-id cd09e02axx -n base_name_srvc8ns001   
az ml service delete realtime -i base_name_srvc8ns001.base_name_srvc8ns001-c88xxx.eastus2  
az ml service create realtime --image-id cd09e02axx -n base_name_srvc8ns002  
az ml service keys realtime -i base_name_srvc8ns002.base_name_srvc8ns002-c888xxx.eastus2  
  
