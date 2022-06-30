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
For use on Kodiak:\
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
Clone this repo:
```bash
git clone https://github.com/akanugan/ticl_hackathon_energyRegression.git
```

The script in /bin will create conda env with all packages need in the present folder:
```bash
bash bin/create-conda-env.sh
```
Then activate the local conda env. by:
```bash
conda activate ./env
```

```bash 
python DataLoader.py
```

To deactivate:
```bash
conda deactivate
```
