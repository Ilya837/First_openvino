import pytest
from ie_classifier import InferenceEngineClassifier
import cv2

model_path = "public\\squeezenet1.1\\FP16\\squeezenet1.1.xml"
weights_path = "public\\squeezenet1.1\\FP16\\squeezenet1.1.bin"
device = 'CPU'

input_image1 = "Image\\37_ru.png" #207
input_image2 = "Image\\4236896-pictures-of-dogs.jpg" #207
input_image3 = "Image\\cat.jpg" #281
input_image4 = "Image\\cat_2.jpg" #173
input_image5 = "Image\\cat_3.jpg" #285
input_image6 = "Image\\image1-1.jpg" #280

img = [input_image1, input_image2,input_image3,input_image4,input_image5,input_image6]

ie_classifier = InferenceEngineClassifier(configPath=model_path, weightsPath=weights_path, device=device)

for i in range(len(img)):
    img[i] = cv2.imread(img[i])


@pytest.mark.parametrize("img,n, index_image,result_image", [(img[0], 1, 0, 207),
                                                             (img, 1, [0,1,2,3,4,5], [207,207,281,173,285,280])])
def test_ie_classifier(img,n, index_image,result_image):
    prob = ie_classifier.classify(img)
    predictions = ie_classifier.get_top(prob, n)
    print(predictions)

    if(type(index_image) == int):
        index_image = [index_image]

    if (type(result_image) == int):
        result_image = [result_image]

    for i in index_image:
        assert predictions[i][0][0] == result_image[i]
