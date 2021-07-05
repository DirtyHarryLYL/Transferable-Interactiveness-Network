# PaStaNet-HOI Benchmark

We resplit [HAKE](http://hake-mvig.cn/home/) and construct a much larger benchmark:
PaStaNet-HOI. 

PaStaNet-HOI provides 110K+ images (77,260 images in train set, 11,298 images in validation set, and 22,156 images in test set). Images are listed in [split_train.txt](split_train.txt), [split_val.txt](split_val.txt) and [split_test.txt](split_test.txt), respectively.

Compared with HICO-DET, PaStaNet-HOI
dataset has much larger train and test sets. The interaction categories are similar to the settings of HICO-DET, but we exclude the 80 “non-interaction” categories and only define 520 HOI categories. This can help to alleviate the annotation missing problem in HICO-DET.
