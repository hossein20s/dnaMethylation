# declare a name for this job to be sample_job
#PBS -N methyl_gpu
# Specify the gpuq queue
#PBS -q gpuq
# Specify the number of gpus
#PBS -l nodes=8:ppn=1:gpus=1
# Specify the gpu feature
#PBS -l feature=gpu
# request 4 hours and 30 minutes of cpu time
#PBS -l walltime=00:10:00
# mail is sent to you when the job starts and when it terminates or aborts
#PBS -m bea
# specify your email address
#PBS -M joshua.j.levy.gr@dartmouth.edu
# Join error and standard output into one file
#PBS -j oe
# By default, PBS scripts execute in your home directory, not the
# directory from which they were submitted. The following line
# places you in the directory from which the job was submitted.
cd $PBS_O_WORKDIR
# run the program
gpuNum=`cat $PBS_GPUFILE | sed -e 's/.*-gpu//g'`
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE=$gpuNum
module load python/3-Anaconda
module load cuda
source activate py36
python embedding.py perform_embedding -lr 1e-4 -wd 0.001 -hlt 500 -n 100 -kl 20 -c
exit 0