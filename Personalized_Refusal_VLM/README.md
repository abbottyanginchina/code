## How to run this code

Please use this command to run all the experiments
```
bash ./scripts/run_xxxx.sh
```

### To obtain activations:
```
python -m experiments.get_activations
```
#### To change datasets, revise 'get_all_datasets' in '.\vti_utils\utils.py'

### To train the model and get the vectors:
```
python -m experiments.steering_network
```

### To generate completions:
```
python -m experiments.generation
```

#### use '--alpha_text' to control the extent of steering

