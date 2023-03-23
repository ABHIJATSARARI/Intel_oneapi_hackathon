import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QComboBox, QCheckBox, QProgressBar, QColorDialog
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weed Detection App")
        
        label = QLabel("Choose a pre-trained model:")
        self.model_combobox = QComboBox()
        self.model_combobox.addItem("model")
        self.model_combobox.addItem("plant_weed")

        self.save_checkbox = QCheckBox("Save Result")

        
        self.threshold_checkbox = QCheckBox("Display Threshold Graph")
        
        self.color_schemes_checkbox = QCheckBox("Use Custom Color Schemes")
        self.color_schemes_checkbox.stateChanged.connect(self.show_color_options)
        
        self.color_options_layout = QVBoxLayout()
        self.color_options_layout.addWidget(QLabel("Box Color:"))
        self.box_color_combobox = QComboBox()
        self.box_color_combobox.addItem("Green")
        self.box_color_combobox.addItem("Blue")
        self.color_options_layout.addWidget(self.box_color_combobox)
        self.color_options_layout.addWidget(QLabel("Label Color:"))
        self.label_color_combobox = QComboBox()
        self.label_color_combobox.addItem("White")
        self.label_color_combobox.addItem("Yellow")
        self.color_options_layout.addWidget(self.label_color_combobox)
        self.color_options_layout.addStretch()
        self.color_options_layout.setContentsMargins(20, 0, 0, 0)
        
        button = QPushButton("Select Image(s)")
        button.clicked.connect(self.select_files)
        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_file)
        
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.model_combobox)
        layout.addWidget(self.threshold_checkbox)
        layout.addWidget(self.color_schemes_checkbox)
        layout.addLayout(self.color_options_layout)
        layout.addWidget(button)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        
    def show_color_options(self, state):
        if state == Qt.Checked:
            self.color_options_layout.setEnabled(True)
        else:
            self.color_options_layout.setEnabled(False)
        
    def select_files(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(None, "Select Image(s)", "", "Image Files (*.png *.jpg *.jpeg)")
        
        box_color = "Green"
        label_color = "White"
        if self.color_schemes_checkbox.isChecked():
            box_color = self.box_color_combobox.currentText()
            label_color = self.label_color_combobox.currentText()
        
        for file_path in file_paths:
            self.detect_weeds(file_path, self.model_combobox.currentText(), self.threshold_checkbox.isChecked(), box_color, label_color)
    
    def save_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(None, "Save Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            image_path = os.path.splitext(file_path)[0] + ".jpg"
            plt.savefig(image_path)
            plt.close()    

    def detect_weeds(self, image_path, model_name, display_threshold, use_custom_colors, box_color, label_color="White"):
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
        
        # Initialize progress bar
        self.progress_bar.setValue(0)
        num_detections = len(outs)
        progress_step = 100 // num_detections
        
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
                    
            # Update progress bar after each set of detections
            self.progress_bar.setValue(self.progress_bar.value() + progress_step)
        
        

        # Define the box and label colors
        if use_custom_colors:
            if box_color == "Green":
                box_color_tuple = (0, 255, 0, 255)  # BGR format
            elif box_color == "Blue":
                box_color_tuple = (255, 0, 0, 255)  # BGR format
            else:
                box_color_tuple = (0, 0, 255, 255)  # BGR format
            if label_color == "White":
                label_color_tuple = (255, 255, 255, 255)  # BGR format
            elif label_color == "Yellow":
                label_color_tuple = (0, 255, 255, 255)  # BGR format
            else:
                label_color_tuple = (0, 0, 0, 255)  # BGR format
        else:
            box_color_tuple = (0, 255, 0, 255)  # default color: green
            label_color_tuple = (255, 255, 255, 255)  # default color: white
    
        # Draw the bounding box and label for each detection
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            color = box_color_tuple
            cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness=2)
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            label_background_color = (0, 0, 0, 255)
            label_text_color = label_color_tuple
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_w, label_h = label_size
            cv2.rectangle(image, (x, y-label_h-10), (x+label_w, y), label_background_color, -1)
            cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
        

        # Option to display threshold output as a graph
        if display_threshold:
            plt.plot(confidences)
            plt.title("Confidence scores")
            plt.xlabel("Detection")
            plt.ylabel("Confidence")
            plt.show()


        # Add code to save the result
        if self.save_checkbox.isChecked():
            # Create a copy of the original image
            result_image = image.copy()

            # Add the bounding boxes and labels to the result image
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                confidence = confidences[i]
                if use_custom_colors:
                    color = box_color_tuple
                else:
                    color = (0, 255, 0, 255)  # Green by default
                cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(result_image, f"{label} {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the result image
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            # Save the result image
            if self.save_checkbox.isChecked():
                self.save_file()

        # Display the original image with the bounding boxes and class labels added to it
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        # Show the image
        cv2.imshow("Weed Detection Result", image)
        cv2.waitKey(0)
        


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

       
