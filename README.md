# Energy Regression (and PID)
----

Repository for our work on energy regression during the ticl hackathon 2022\
Use this on patatrack 

To run the Singularity:
```bash
export SINGULARITY_CACHEDIR=$WORK/
export SINGULARITY_BIND="/eos"
singularity shell --nv /eos/cms/store/user/bmaier/sandboxes/geometricdl.sif 
```
For use on Kodiak:
Assuming that anaconda is already setup, you can check this by:
```bash
conda activate
```
after sourcing the env. if already present in bash script like:
``` bash
__conda_setup="$('/home/$USER/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/$USER/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/$USER/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/$USER/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
```
Create a new conda env with python pre-installed python3:

```
conda create --name ticl_energyReg python=3.7
conda activate ticl_energyReg
```

Note: it is necessary to execute 'conda activate ticl_energyReg' at the start of logging in
To deactivate the virtual environment, simply:

```
conda deactivate
```
