import os
import time

import cv2
import sys
import argparse
import numpy as np
import logging as log
from openvino.inference_engine import IECore, IENetwork
import threading
from threading import Thread
import copy

from IPython.display import Image

os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\deployment_tools\\ngraph\\lib")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\deployment_tools\\inference_engine"
                     "\\external\\tbb\\bin")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\deployment_tools\\inference_engine\\bin"
                     "\\intel64\\Release")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\deployment_tools\\inference_engine"
                     "\\external\\hddl\\bin")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\opencv\\bin")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\python\\python3.8\\openvino\\libs")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.4.689\\deployment_tools\\inference_engine\\include")


class InferenceEngineClassifier:
    def __init__(self, configPath=None, weightsPath=None, device='CPU', classesPath=None):
        self.ie = IECore()
        self.net = self.ie.read_network(model=configPath, weights=weightsPath)
        self.exec_net = self.ie.load_network(network=self.net, device_name=device, num_requests = 3)
        if classesPath:
            self.classes = [line.rstrip('\n') for line in open(classesPath)]

        pass

    def get_top(self, prob, topN=1):
        if (isinstance(prob, np.ndarray )): #нужно заменить тип
            prob = [prob]
        best = []
        res = []
        for i in range(len(prob)):
            prob[i] = np.squeeze(prob[i])
            best.append(np.argsort(prob[i])[-topN:])
            result = []
            for j in range(len(best[i]) - 1, -1, -1):
                try:
                    classname = self.classes[int(best[i][j])]
                except:
                    classname = best[i][j]
                line = [classname, prob[i][best[i][j]] * 100.0]
                result.append(line)
            res.append(result)

        return res

    def _prepare_image(self, image, h, w):
        if image.shape[:-1] != (h, w):
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        return image
        pass

    def classify_2(self, image, h, w, input_blob, out_blob):
        output =[]
        for i in range(len(image)):
            blob = self._prepare_image(image[i], h, w)
            out = self.exec_net.infer(inputs={input_blob: blob})
            output.append(out[out_blob])
        return output

    def classify_3(self, image, h, w, input_blob, out_blob):
        for i in range(len(image)):
            image[i] = self._prepare_image(image[i],h,w)

        for request_id in range(len(image)):
            self.exec_net.start_async(input_data=image[request_id], request_id=request_id)

        for i in range(self.requets_counter):
            self.exec_net.requests[i].wait(-1)

        output = [copy(self.exec_net.requests[i].outputs) for i in range(len(image))]

        return output


    def classify(self, image):
        input_blob = next(iter(self.net.input_info))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.input_info[input_blob].input_data.shape

        if (isinstance(image, np.ndarray)):
            image = [image]

        return self.classify_2(image,h,w,input_blob,out_blob)
        pass
