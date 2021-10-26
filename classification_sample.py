from ie_classifier import InferenceEngineClassifier
import cv2


model_path = "public\\squeezenet1.1\\FP16\\squeezenet1.1.xml"
weights_path = "public\\squeezenet1.1\\FP16\\squeezenet1.1.bin"
device = 'CPU'

input_image1 = "Image\\37_ru.png"
input_image2 = "Image\\4236896-pictures-of-dogs.jpg"

ie_classifier = InferenceEngineClassifier(configPath=model_path, weightsPath=weights_path, device=device)
input_image1 = cv2.imread(input_image1)
input_image2 = cv2.imread(input_image2)
img = [input_image1, input_image2]
prob = ie_classifier.classify(input_image1)
predictions = ie_classifier.get_top(prob, 2)
print(predictions)
