# MRDB
This repository contains the full training/test code in PyTorch and a trained model for the WACV paper "Enhanced generative adversarial network for 3Dbrain MRI super-resolution". Please cite the paper if you found this code useful for your research. 

Currently this only contains the test script. To run the test please follow the steps. 
1. Please download and unzip the lowres, highres, mask_cache and RRDB_G64_nobn folders to the base folder.
https://drive.google.com/drive/folders/16rE6HgPZ2I0pfvSO15ujio_kIdCte9qb?usp=sharing

Also create an empty folder name evalout in the base folder.

2. Change the GPU/directory in evaluate.py if the above downloaded folder is not placed under the base folder.

3. run evaluate.py

4. One should see a output in the ./evalout folder. One can open it and compare it against the gt highres.

The training script and the PPD discriminator will be uploaded soon.
