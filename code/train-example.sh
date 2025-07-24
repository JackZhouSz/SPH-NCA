# Below configurations are tested on NVIDIA RTX 2080 (8GB)
# Change the `gpu_index` to your GPU index if you have multiple GPUs.
# To match VRAM limitations, change the following parameters:
# - reduce `image_size` (#pixels), and perhaps increase `h` (SPH kernel radius), if the quality degrades due to low pixel density
# - reduce `steps_range` (NCA steps), although the current configuration is pretty minimal

# RGBA Image (Gecko)
python train.py --gpu_index 0 --seed 1 --target 🦎 --lr 3e-3 \
    --loss mse_simple --loss_weight_overflow 0.05 \
    --initial_feature radial --initial_feature_radius 0.16 \
    --use_alpha true --alpha_premultiply true --wrap false \
    --h 0.1 --image_size 100 --target_size 75 --training_iter 8000 \
    --steps_range 32,48 --steps_increment 10 --nca_update gated

# # Exemplar-guided (Zebra) 
# python train.py --gpu_index 0 --seed 1 --img ./yarn.jpg --lr 3e-3 \
#     --loss ot --loss_weight_style 1 --loss_weight_color 0.05 --loss_weight_overflow 0.05 \
#     --initial_feature random --use_alpha false --wrap true \
#     --h 0.1 --image_size 64 --target_size 64 --training_iter 8000 \
#     --steps_range 16,24 --steps_increment 10 --nca_update gated

# # Text-guided (Jellybeans)
# python train.py --gpu_index 0 --seed 1 --lr 3e-4 \
#     --loss clip_multiscale --loss_weight_clip 1 --loss_weight_overflow 0.05 \
#     --initial_feature random  --use_alpha false --wrap true  \
#     --h 0.1 --image_size 64 --target_size 64 --training_iter 8000 \
#     --steps_range 16,24 --steps_increment 10  --nca_update gated \
#     --clip_guide "a colorful pile of red, yellow, green and blue jellybeans" \
#     --clip_multiscale_scales 2,1,0.5
