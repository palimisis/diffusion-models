#/bin/bash

# Instructions: Call without arguments to start training from the
# beginning. Call with a single argument which will be the checkpoint
# file to start from a previously saved checkpoint.
#
# If checkfile.chk is the checkpoint filename, then number of
# completed epochs is stored in a file named checkfile.numepochs
#
# Please define the `EXP_NAME` variable in order to correspond
# to the name of the running experiment.
#
# The N_STEPS variable is the number of times the experiment will be run.
# The EPOCHS_PER_STEP is the epochs that the model will be trained in each
# experiment. Both the EPOCHS_PER_STEP var is passed to the .py script
# as an argument. The example .py script handles the rest.
#
# For example: If I want to train my model for 30 epochs and
# train my model for 3 epochs each run, I will set:
# N_STEPS = 10 , EPOCHS_PER_STEP=3

N_STEPS=100
EPOCHS_PER_STEP=1
EXP_NAME="sixray_small"
CHK_PREFIX="/home/$(whoami)/experiments/${EXP_NAME}"
DATE=`date +"%s"`
LOGDIR="${CHK_PREFIX}/${DATE}/logs"
mkdir -p "${CHK_PREFIX}/${DATE}"
mkdir -p "${LOGDIR}"
epochs=0
CHK_NAME="${EXP_NAME}.chk"
jid=${EXP_NAME}_0
STDOUT="${LOGDIR}/$jid"

if [ -z $1 ]; then
    CHK_FILE="${CHK_PREFIX}/${DATE}"
    CHK_EPOCHS="${CHK_PREFIX}/${DATE}/${CHK_NAME}"
    sbatch -J $jid -o "${STDOUT}.out" -e "${STDOUT}.err" /home/$(whoami)/diffusion-models/scripts/job.sh --epochs $EPOCHS_PER_STEP --cpt_path ${CHK_FILE} --dataset_path=/home/it21902/datasets/sixray/dataset/JPEGImage --dataset=sixray --image_size=128 --batch_size=16
    echo $EPOCHS_PER_STEP > ${CHK_EPOCHS}.numepochs
else
    echo "Got into else"
    CHK_FILE=$1
    epochs=`cat ${CHK_EPOCHS}.numepochs`
    tot_epochs=$epochs+$EPOCHS_PER_STEP
    sbatch -J $jid -o "${STDOUT}.out" -e "${STDOUT}.err" /home/$(whoami)/diffusion-models/scripts/job.sh --epochs $tot_epochs --cpt_path ${CHK_FILE}  --dataset_path=/home/it21902/datasets/sixray/dataset/JPEGImage --dataset=sixray --image_size=128 --batch_size=16
    echo $tot_epochs > ${CHK_EPOCHS}.numepochs
fi


for ((i=1; i<${N_STEPS}; i++)); do
    epochs=`cat ${CHK_EPOCHS}.numepochs` # $epochs+$EPOCHS_PER_STEP
    let tot_epochs=$epochs+$EPOCHS_PER_STEP
    let d=$i-1
    jid=${EXP_NAME}_$i
    STDOUT="${LOGDIR}/$jid"
    depends=$(squeue --noheader --format %i --name ${EXP_NAME}_${d})
    sbatch -J $jid -o "${STDOUT}.out" -e "${STDOUT}.err" -d afterany:${depends} /home/$(whoami)/diffusion-models/scripts/job.sh --epochs $tot_epochs --cpt_path ${CHK_FILE}  --dataset_path=/home/it21902/datasets/sixray/dataset/JPEGImage --dataset=sixray --image_size=128 --batch_size=16
    echo $tot_epochs > ${CHK_EPOCHS}.numepochs
done

