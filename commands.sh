sbatch -J $jid "${STDOUT}.out" -e "${STDOUT}.err" /home/it21902/diffusion-models/scripts/simple_job.sh \
--dataset_path=/home/it21902/datasets/sixray2/dataset/JPEGImage \
--dataset=sixray \
--image_size=1024 \
--cpt_path=/home/it21902/diffusion-models/sixray-demo \
--epochs=5

python train.py --dataset_path=/home/it21902/datasets/sixray2/dataset/JPEGImage \
--dataset=sixray \
--image_size=1024 \
--cpt_path=/home/it21902/diffusion-models/sixray-demo \
--epochs=5