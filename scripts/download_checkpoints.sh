set -e  # exit when any command fails
mkdir -p pretrained/
cd pretrained/
gdown --id "1E4MPrlndyi1I0It9FTighfnz4fyU39lY&confirm=t" -O photo_wct.pth # download weights for style transfer module
gdown --id "1Y602qa4VSQJB2dXppJV7vXD_m4nAjJqj&confirm=t" -O resnet50_v1c-2cccc1ad.pth # download ResNet-50 ImageNet pre-trained weight
gdown --id "1gK_fEWL80hEww6W_UBbDuViPz6Sx9ZtI&confirm=t" -O resnet101_v1c-e67eebb6.pth # download ResNet-101 ImageNet pre-trained weights
cd ../

mkdir -p work_dirs/
cd work_dirs/
gdown --id "1dh0YeOneAc7s7ug1rV0ZhQcrrW5ZgbaU&confirm=t"  # TLDR on GTA->Cityscapes
tar -xvf 230203_0112_iter_40000_lr_3e-05_orig_0.5_style_0.5_regw_0.005_regr_1.0_disentw_0.005_disentr_1.0_threshold_0.1_seed_300_fd325.tar
rm 230203_0112_iter_40000_lr_3e-05_orig_0.5_style_0.5_regw_0.005_regr_1.0_disentw_0.005_disentr_1.0_threshold_0.1_seed_300_fd325.tar
cd ../
