# Rank-Preserving Surrogate Model
To speed up Neural Architecture Search (NAS) algorithms, several existing approaches use surrogate models that predict the neural architectures' precision instead of training each sampled one. However, these approaches do not preserve the ranking between different architectures. This repository includes the code for RS-NAS. Surrogate models trained specifically to preserve the ranking and score the architectures for NAS. Our code is heavily inspired by NAS-Bench-301 and we include its code as a submodule to execute the tests and experimentations. 

Weights of each model will be available : [RS_NAS_Models_1.0.zip](https://figshare.com/articles/dataset/RS_NAS_models_1_0/14134847)
The code isn't stable yet, more versions are yet to come. 

# Surrogate Models 
We tested 5 different surrogate models: 
* MLP 
* GCN
* GIN
* LGBoost 
* XGBoost 

For each one of them, there's a separate training script with the dedicated hyperparameters. You can find the relative code in the folder '/surrogate_models/$MODEL_NAME' 

# Add Benchmark files 
Include these downloaded file in a folder 'nas_benchmark' at the root of the project directory. 
``` 
wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
wget https://ndownloader.figshare.com/files/25506206?private_link=7d47bf57803227af4909 -O NAS-Bench-201-v1_0-e61699.pth
wget https://ndownloader.figshare.com/files/24693026 -O nasbench301_models_v0.9.zip
unzip nasbench301_models_v0.9.zip
```
