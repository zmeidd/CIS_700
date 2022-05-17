# CIS_700
### Download the HolStep Dataset
```
mkdir data/raw_data && cd data/raw_data
wget http://cl-informatik.uibk.ac.at/cek/holstep/holstep.tgz
tar -xvzf holstep.tgz
```

### Replace the HolStep directory in deal.py
```
replace  "source_dir = 'D:\Download\holstep'" in main function to 
your downloaded holstep folder
```

### Run deal.py to get the preprossed numpy files
```
python3 deal.py
```


### Run main.py to derive the results and plots
```
python3 main.py
```


