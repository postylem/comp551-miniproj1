# comp551-miniproj1

## Sept-17 notes:
Tasks to do before thursday-ish:
- Sam, Lambert: do preprocessing on the two datasets.
- Jacob: make a skeleton for the model.py file.

## Sept-24
I have printed out some reporting to a log file, for record-keeping. --j
for instance, replacing density with density^2  didn't really change much:

wide dataset
with density (original)
```
Mean error across folds:     0.2776704545454545
Mean accuracy across folds:  0.7223295454545454
Mean precision across folds: 0.7641294587506356
Mean recall across folds:    0.6565415768219799
```
with density2 (squared density)
```
Mean error across folds:     0.2770572100313479
Mean accuracy across folds:  0.7229427899686521
Mean precision across folds: 0.7680212662505879
Mean recall across folds:    0.6721995337510766
```
density2 - density 
```
difference in accuracy across folds:  0.0006132445141
difference in precision across folds: 0.0038918075
difference in recall across folds:    0.01565795693
```
a (very) small improvement.
