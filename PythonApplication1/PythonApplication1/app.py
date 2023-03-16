import cv2
import numpy as np
from openvino.inference_engine import IECore
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setObjectName("startButton")
        self.verticalLayout.addWidget(self.startButton)
        self.quitButton = QtWidgets.QPushButton(self.centralwidget)
        self.quitButton.setObjectName("quitButton")
        self.verticalLayout.addWidget(self.quitButton)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Object Detection App"))
        self.label.setText(_translate("MainWindow", "Click Start to Begin"))
        self.startButton.setText(_translate("MainWindow", "Start"))
        self.quitButton.setText(_translate("MainWindow", "Quit"))

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.startButton.clicked.connect(self.start)
        self.quitButton.clicked.connect(self.quit)

    def start(self):
        # Load the YOLO model using the OpenVINO Inference Engine
        model_xml = "crop_weed.cfg"
        model_bin = "crop_weed_detection.weights"
        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)
        input_blob = next(iter(net.input_info))
        out_blob = next(iter(net.outputs))
        exec_net = ie.load_network(network=net, device_name="CPU")

        # Load the class names associated with the YOLO model
        class_names_file = "obj.names"
        with open(class_names_file, "r") as f:
            class_names = [cname.strip() for cname in f.readlines()]

        # Define the function to perform object detection on an input image
        def detect_objects(image):
            # Preprocess the input image for the YOLO model
            input_shape = net.input_info[input_blob].input_data.shape
            resized_image = cv2.resize(image, (input_shape[3], input_shape[2]))
            input_data = resized_image.transpose((2, 0, 1))
            input_data = input_data.reshape((1,) + input_data.shape)
            input_data = (input_data - 127.5) / 127.5  # normalize pixel values to [-1, 1]

            # Run the YOLO model on the input image
            outputs = exec_net.infer(inputs={input_blob: input_data})
            output_blob = next(iter(outputs))
            output = outputs[output_blob]

            # Post-process the YOLO model output
            boxes, confidences, class_ids = [], [], []
            for detection in output[0, 0]:
                confidence = detection[2]
                if confidence > 0.5:  # only keep detections with confidence > 0.5
                    class_id = int(detection[1])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    box = detection[3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    box = box.astype("int")
                    boxes.append(box)

            # Draw the bounding boxes and class labels on the input image
            colors = np.random.uniform(0, 255, size=(len(class_names), 3))
            for i, box in enumerate(boxes):
                color = colors[class_ids[i]]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                label = "{}: {:.2f}".format(class_names[class_ids[i]], confidences[i])
                cv2.putText(image, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the output image in the UI
            output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = output_image.shape
            bytes_per_line = ch * w
            q_image = QtGui.QImage(output_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(q_image))

    def quit(self):
        self.close()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())

