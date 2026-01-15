
# Internal README.md

### To create the `botbuster-py38` env
Note that you MUST do the install this way or it will fail on MacOS due to the change in Mac Silicon requirements for numpy that happened around when this numpy version came out.
```
conda create -n botbuster-py38 python=3.8 pip ipykernel
conda activate botbuster-py38
conda install -c conda-forge "numpy==1.23.1" "pandas==1.1.3" "scikit-learn==0.24.2"
pip install --no-deps -r requirements.txt
pip install colorama
```
*Note that pickled models were trained with sklearn 0.24.2; don’t upgrade sklearn unless you also retrain/export models.*

### How to run `botbuster_reddit.py`

```
# activate botbuster-py38 if you haven't already
cd BotBuster
python botbuster_reddit.py
```

### To save current env requirements
Note that this will not work with a simple `pip install requirements.txt` later, see above notes about the `numpy` version issues on MacOS. 
```
pip install pip-chill && pip-chill > requirements.txt
```