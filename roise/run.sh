python /home/zheng/zheng/rosie_reproduce/roise_reorganize/convert_to_zarr_memory.py \
--input-dir '/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter' \
--output-dir '/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores_Zarr' \
--chunk-height 512 \
--chunk-width 512 \
--chunk-channels 3 \
--markers HE CD3 CD8 CD56 CD68 CD163 MHC1 PDL1 \
--max-images 20


# python dataframe_generator.py \
# --tma-dir '/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores' \
# --output 'tma_data.parquet'

# # Load dataframe with streaming for large files
# python patch_dataset.py \
# --parquet '/home/zheng/zheng/rosie_reproduce/roise_reorganize/tma_data.parquet' \
# --image-root '/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores' \
# --streaming \
# --max-samples 1000

cd /home/zheng/zheng/rosie_reproduce/roise

# python train.py \
#     --root '/home/zheng/zheng/rosie_reproduce' \
#     --data-file '/home/zheng/zheng/rosie_reproduce/roise/datalabel_generator/tma_data.parquet' \
#     --target-biomarkers CD3 \
#     --batch-size 32 \
#     --lr 1e-4