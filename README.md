# image_classifier
Udacity course project Image Classifier
Python debug:
The two .ipynb files are identical except for the transform.ToTensor vs. transform.ToPILImage in the train_transform lines.
(I tried other tensor- and PIL-related transforms, and commenting them out altogether, with similar results.)

Image_Classifier_ToTensor.ipynb gives error:
      TypeError: img should be PIL Image. Got <class 'torch.Tensor'>
      
Image_Classifier_ToPILImage.ipynb gives error:
      TypeError: pic should be Tensor or ndarray. Got <class 'PIL.Image.Image'>.
      
