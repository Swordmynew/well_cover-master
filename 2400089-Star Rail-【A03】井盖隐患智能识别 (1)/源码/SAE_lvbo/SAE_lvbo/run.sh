export CUDA_VISIBLE_DEVICES=0,1
source activate DGNet

python MyTrain.py --gpu_id 0,1  --save_path snapshot/ --batchsize 16 --model EF-B4 --train_root "/home/user_kkj/data1/kangkejun/TrainDataset/" --val_root "/home/user_kkj/data1/kangkejun/TestDataset/CAMO/" --trainsize 352 --load /home/user_kkj/SAE/snapshot1/Net_epoch_best-EF-B4.pth
#TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=4 --use_env MyTrain.py --gpu_id 0,1,2,3  --save_path /home/user_kkj/data1/kangkejun/shouNet/snapshot/Exp-DGNet/ --batchsize 3 --model res50 --train_root /home/user_kkj/data1/kangkejun/TrainDataset/ --val_root /home/user_kkj/data1/kangkejun/TestDataset/CAMO/ --trainsize 352
#--load /home/user_kkj/data1/kangkejun/shouNet/snapshot/Exp-DGNet/Net_epoch_best.pth

# python MyTrain.py --gpu_id 2  --save_path snapshot/ --batchsize 10 --model PVTv2-B4 --train_root Dataset/TrainDataset/ --val_root Dataset/TestDataset/CAMO/
 
