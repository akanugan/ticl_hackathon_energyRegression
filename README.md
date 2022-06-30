# Energy Regression (and PID)
----

Repository for our work on energy regression during the ticl hackathon 2022\
-------------------------------
Use this on patatrack 
To run the Singularity:
```bash
export SINGULARITY_CACHEDIR=$WORK/
export SINGULARITY_BIND="/eos"
singularity shell --nv /eos/cms/store/user/bmaier/sandboxes/geometricdl.sif 
```
-------------------------------

# For use on Kodiak:
Assuming that anaconda is already setup, you can check this by:
```bash
conda activate
```

Clone this repo:
```bash
git clone https://github.com/akanugan/ticl_hackathon_energyRegression.git
cd ticl_hackathon_energyRegression/
git checkout master
```

The script in /bin will create conda env with all packages need in the present folder:
```bash
bash bin/create-conda-env.sh
```
Then activate the local conda env. by:
```bash
conda activate ./env
```

Now run the scripts for the data loader and testing the model:
```bash 
python DataLoader.py
python models.py
```

To deactivate:
```bash
conda deactivate
```
