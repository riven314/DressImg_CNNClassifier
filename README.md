# Dress Image Classifier
Using fastai package, I am training a deep learning classifier for dress images of more than 200 classes.

# Summary
For data, training the full images of 200 classes is way more difficult and computationally costly, so I try to start off with a smaller subset of the full images:
1. 5 classes (~ 12500 images)
2. 30 classes (~ 75000 images)
3. Full classes (~ 560000 images)

For model architecture, I tried out the following models in order:
1. Pretrained resnet34 (with randomized fully connected layers)
2. Pretrained resnext50 (with randomized fully connected layers)
3. Resnet56 from scratch
4. Pretrained resnext101 (with randomized fully connected layers)

I implemented different techniques on fine-tuning parameters - such as finding a suitable learning rate and dropout rate, implement data augmentation, cosine annealing and test time augmentation. 
