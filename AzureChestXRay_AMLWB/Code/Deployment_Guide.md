Install Azure Machine Learning Workbench AMLWB (https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation)   
1. Set up your environment:   
  Open an amlwb cli and follow this [guide](https://docs.microsoft.com/en-us/azure/machine-learning/preview/tutorial-classifying-iris-part-2#execute-scripts-in-the-azure-machine-learning-cli-window):        
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
  -->Deploy a linux VM (e.g. a linux [DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro))      
  -->For best results, use a a deep learning linux VM (https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning).   
  -->You may need a [GPU VM](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu) for training, and a second CPU VM for testing operationalization.   
     
  --> Make sure the vm fqdn is set up. The disk sizes must cover data requirements. Default data disks do not survive machine reboot events. To keep the data available between machine reboots, either make the OS disk larger, or attach an external Azure VHD.    
     
  --> use https://docs.microsoft.com/en-us/azure/machine-learning/preview/known-issues-and-troubleshooting-guide#vm-disk-is-full to resize the VM disk if needed   
    #Deallocate VM (stopping will not work)   
    $ az vm deallocate --resource-group myResourceGroup  --name myVM   
    # Update Disc Size   
    $ az disk update --resource-group myResourceGroup --name myVM --size-gb 250   
    # Start VM       
    $ az vm start --resource-group myResourceGroup  --name myVM   
     
  --> ssh into the remote VM and create the folder structure that will be used by AMLWB to map host locations to directories in running AMLWB containers   
  loginuser@deeplearninggpuvm:~$ sudo mkdir -p /datadrive01/amlwbShare   
  loginuser@deeplearninggpuvm:~$ sudo chmod ugo=rwx /datadrive01/amlwbShare/   
  loginuser@deeplearninggpuvm2:~$ ls -l /datadrive01/   
  total 4   
  drwxrwxrwx 2 root root 4096 Feb  5 18:33 amlwbShare   
     
  2.2 Get the NIH Chext Xray images   
  Go to https://nihcc.app.box.com/v/ChestXray-NIHCC   
  Store the images from images (https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737) dir as unarchived files in a blob storage account.    
  They will be downloaded later in Code\01_DataPrep\_001_get_data.ipynb, and will land into a dir on the remote VM created above:   
  loginuser@deeplearninggpuvm2:~$ ls /datadrive01/amlwbShare/crt_ea/grt_work_space/crt_experiment/chestxray/data/ChestX-ray8/ChestXray-NIHCC/ | head -2   
  00000001_000.png   
  00000001_001.png   
   
  2.3 Get patient-image map file    
  Download manually NIH data file Data_Entry_2017.csv (https://nihcc.app.box.com/v/ChestXray-NIHCC) into this dir on the remote VM created above:   
  loginuser@deeplearninggpuvm2:~$ ls -l /datadrive01/amlwbShare/crt_ea/grt_work_space/crt_experiment/chestxray/data/ChestX-ray8/ChestXray-NIHCC_other   
  total 7680   
  -rw-rw-r-- 1 loginvm0011 loginvm0011 7861152 Feb  7 02:54 Data_Entry_2017.csv   
  Data_Entry_2017.csv is the patients to images map and will be used by \Code\02_Model\000_preprocess.ipynb to create the train/validate/test partitions.   
   
     
  2.4      
  -->in AMLWB cli, create AMLWB compute context:      
  az ml computetarget attach remotedocker --name <compute_context_name> --address <dsvm_fqdn> --username <dsvm_login> --password <dsvm_password>        
  the command above will create \aml_config\<compute_context_name>.runconfig and \aml_config\<compute_context_name>.compute files that control the AMLWB compute contexts   
   
  -->Check the existing compute targets:        
  az ml computetarget list      
        
  For CPU compute contexts:      
  -->edit <compute_context_name>.runconfig file:      
        CondaDependenciesFile: aml_config/conda_dependencies_o16n.yml      
  	Framework: Python        
        PrepareEnvironment: true        
  -->edit <compute_context_name>.compute file:        
        baseDockerImage: georgedockeraccount/utils_with_amlwb_base_cpu:azcopyenabled      
	nativeSharedDirectory: /data/datadrive01/amlwbShare/        
	sharedVolumes: true        
	      
  For GPU compute contexts:      
  -->edit <compute_context_name>.runconfig file:      
        CondaDependenciesFile: aml_config/conda_dependencies_gpu.yml      
  	Framework: Python        
        PrepareEnvironment: true        
  -->edit <compute_context_name>.compute file:        
	baseDockerImage: georgedockeraccount/utils_with_amlwb_base_gpu:azcopyenabled      
	nativeSharedDirectory: /data/datadrive01/amlwbShare/      
	nvidiaDocker: true   
	sharedVolumes: true        
          
  -->go back to cli:        
  az ml experiment prepare -c <compute_context_name>        
  -> while the preparation is running, you can check on linux host machine how docker is running:        
  	sudo docker images        
  	sudo docker ps -a        
   
E.g.:   
loginuser@deeplearninggpuvm:~$ sudo docker images   
REPOSITORY                                      TAG                 IMAGE ID            CREATED             SIZE   
azureml_88865f7583e9e1fd502a32a7717aa1f0        latest              a35a05a9b295        16 minutes ago      7.21GB   
georgedockeraccount/utils_with_amlwb_base_gpu   azcopyenabled       2e6da7a1351c        4 weeks ago         3.89GB   
   
  -> see \Code\docker\Dockerfile for details on how the docker images have been created.   
   
   
3. Run experiments: 	   
Code\01_DataPrep\_GetData.ipynb   
\Code\02_Model\000_preprocess.ipynb : creates the train/validate/test partitions and saves all NIH chest XRay images as numpy objects on disk   
\Code\02_Model\010_train.ipynb : trains a densenet model (pretrained on imagenet) on NIH chest xray data   
\Code\02_Model\040_cam_simple.ipynb : shows CAM visualizations  
