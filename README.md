# CELEBA RAM

This is actually a impl of ram on celeba classification
does classification seperately for each label

train semi-supervised with:
```
python main.py --dataset celeba --image_size 224 --num_classes 2 --selected_attrs 'Male' --patch_size 16 --is_train True --supervised_attention_prob 0.2
```

test results with 
```
python main.py --dataset celeba --image_size 224 --num_classes 2 --selected_attrs 'Male' --patch_size 16 --is_train False
```
