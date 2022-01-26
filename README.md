# image_classifier
Udacity course project Image Classifier

The Imsge_Classifier.part1_save2.ipynb show the new classifier with an error message in the training cell, ie. that the image has fewer dimensions than expected.

The image_classifier_part1.ipynb file shows a bug / mismatch between the VGG16 model and the classifier for 3x224x224 inputs images.



Python debug (an earlier problem I had with transforms):
The two other .ipynb files are identical except for the transform.ToTensor vs. transform.ToPILImage in the train_transform lines.
(I tried other tensor- and PIL-related transforms, and commenting them out altogether, with similar results.)

Image_Classifier_ToTensor.ipynb gives error:
      TypeError: img should be PIL Image. Got <class 'torch.Tensor'>
      
Image_Classifier_ToPILImage.ipynb gives error:
      TypeError: pic should be Tensor or ndarray. Got <class 'PIL.Image.Image'>.
      
