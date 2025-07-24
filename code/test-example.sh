# # Test checkpoint on image
python test.py --gpu_index 0 --seed 1 --checkpoint $1 \
    --initial_feature random  --use_alpha false --wrap true \
    --h 0.1 --image_size 64 \
    --steps 128 --nca_update gated \
    --output_dir ./output/

# # Test checkpoint on mesh
# python test.py --gpu_index 0 --seed 1 --checkpoint $1 \
#     --initial_feature random  --use_alpha false --wrap true \
#     --h 0.1 --surface ./data/bunny.obj --surface_scale 1.5 \
#     --steps 128 --nca_update gated \
#     --output_dir ./output/
