python compute_metrices.py --ckpt color --dataset-dir dataset/edges2handbags/train_right_images_64 --sample-dir samples_nfe999

python evaluations/evaluator.py ../data/edges2handbags_ref_64_data.npz  ../results/color/samples_nfe999/recon.pt --metric fid


python evaluations/evaluator.py results/normal/samples_nfe999/converted_samples/diode_ref_256_data.npz  results/normal/samples_nfe999/converted_samples/samples_16502x256x256x3.npz --metric fid
python evaluations/evaluator.py results/normal/samples_nfe999/converted_samples/diode_ref_256_data.npz  results/normal/samples_nfe999/converted_samples/samples_16502x256x256x3.npz --metric lpips

python evaluations/evaluator.py results/normal/samples_nfe300/clean_img_fid.npz  results/normal/samples_nfe200/converted_samples/samples_16502x256x256x3.npz --metric fid
python evaluations/evaluator.py results/normal/samples_nfe20/clean_img_fid.npz  results/normal/samples_nfe20/recon_img_fid.npz --metric fid
python evaluations/evaluator.py clean_img_fid.npz  recon_img_fid.npz --metric fid