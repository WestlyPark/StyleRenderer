# StyleGAR
# TODO: add arxiv link
Implementation of Inverting Generative Adversarial Renderer for Face Reconstruction 

# TODO: for test
Currently, some models are being modified with open-code resources (3dmm, landmark, segmentation), to get rid of commercial models, will update soon.

## Usage
First align all faces according to landmarks:

> python uitls_face.py --lmk dlib --bfm BFM.mat --output OUTPUT_PATH DATASET_PATH

This will align according to specified landmark model, you can change to other kinds of landmark detector

Create lmdb datasets:

> python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... OUTPUT_PATH

This will convert images to jpeg and pre-resizes it. This implementation does not use progressive growing, but you can create multiple resolution datasets using size arguments with comma separated lists, for the cases that you want to try another resolutions later.

Then you can train model in distributed settings

> python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE LMDB_PATH

train.py supports Weights & Biases logging. If you want to use it, add --wandb arguments to the script.

### Generate samples

> python generate.py --sample N_FACES --pics N_PICS --ckpt PATH_CHECKPOINT

You should change your size (--size 256 for example) if you train with another dimension.

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid

The work is built on the implementation of stylegan2 with Pytorch (https://github.com/rosinality/stylegan2-pytorch.git)
