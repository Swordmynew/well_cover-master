1.Download the datasets and set them in the direction Dataset. You can download them from links made public by other authors as: downloading the testing dataset and moving it into Dataset/TestDataset/, which can be found in https://drive.google.com/file/d/1L4zo8Mml08Q2sDPnqT01Nqxx4wv6FMDa/view?usp=sharing. 
downloading the training dataset and moving it into Dataset/TrainDataset/, which can be found in https://drive.google.com/file/d/11-5bBnfVal03D74dtRlJUpuWfmVLc8x9/view?usp=sharing.

2. conda create -n SAE python=3.6 
   conda activate SAE
   conda install --yes --file requirement.txt

3. bash run.sh for train and bash test.sh for test, after test you need bash eval.sh to eval the metrics.