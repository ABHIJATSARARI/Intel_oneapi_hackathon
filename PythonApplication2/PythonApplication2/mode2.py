import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QComboBox, QCheckBox
import matplotlib.pyplot as plt



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weed Detection App")
        
        label = QLabel("Choose a pre-trained model:")
        self.model_combobox = QComboBox()
        self.model_combobox.addItem("model")
        self.model_combobox.addItem("plant_weed")
        
        self.threshold_checkbox = QCheckBox("Display Threshold Graph")
        
        button = QPushButton("Select Image")
        button.clicked.connect(self.select_file)
        
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.model_combobox)
        layout.addWidget(self.threshold_checkbox)
        layout.addWidget(button)
        
        self.setLayout(layout)
        
    def select_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        detect_weeds(file_path, self.model_combobox.currentText(), self.threshold_checkbox.isChecked())
    

def detect_weeds(image_path, model_name, display_threshold):
    # Load the YOLOv3 network
    net = cv2.dnn.readNet(f"{model_name}.cfg", f"{model_name}_detection.weights")
    
    # Load the list of object names
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Create a blob from the image and run it through the network
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    
    # Parse the outputs and get the bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w//2
                y = center_y - h//2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Draw the bounding boxes and labels on the image
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
    
    # Display the image
    cv2.imshow("Weed Detection", image)

    # Option to display threshold output as a graph
    if display_threshold:
        plt.plot(confidences)
        plt.title("Confidence scores")
        plt.xlabel("Detection")
        plt.ylabel("Confidence")
        plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows() 


    
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

