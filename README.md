# Flower Image Classifier

## Command Line Application Usage

### Train
Train a new network on a data set with train.py  <br />
Basic usage: python train.py data_directory  <br />
Prints out training loss, validation loss, and validation accuracy as the network trains  <br />
#### Options: 
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory <br />
Choose architecture: python train.py data_dir --arch "vgg16"<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VGG16 or DenseNet121 architecture is available<br />
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 <br />
Use GPU for training: python train.py data_dir --gpu <br />

### Predict
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability. <br />
Basic usage: python predict.py /path/to/image checkpoint <br />
#### Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3 <br />
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json <br />
Use GPU for inference: python predict.py input checkpoint --gpu <br />
