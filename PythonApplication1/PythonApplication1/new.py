import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

# Load the YOLO model
model = cv2.dnn.readNetFromDarknet('crop_weed.cfg', 'crop_weed_detection.weights')
with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set up the app UI
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Weed Detection')
        self.setGeometry(100, 100, 800, 600)
        self.image = None
        self.result_image = None
        self.initUI()

    def initUI(self):
        # Add a menu bar with a file menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        open_file_action = file_menu.addAction('Open Image')
        open_file_action.triggered.connect(self.open_file)

        # Add an image view widget to display the input and output images
        self.image_view = QLabel(self)
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setGeometry(10, 10, 780, 480)

        # Add a button to start the detection process
        detect_button = QPushButton('Detect Weeds', self)
        detect_button.setGeometry(10, 500, 150, 50)
        detect_button.clicked.connect(self.detect_weeds)

        # Add a progress bar to indicate the detection progress
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(180, 515, 600, 20)
        self.progress_bar.setVisible(False)

        self.show()

    def open_file(self):
        # Open a file dialog to select an image file
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.xpm *.jpg *.bmp)')
        if file_name:
            self.image = cv2.imread(file_name)
            self.result_image = self.image.copy()
            self.display_image(self.image)

    def detect_weeds(self):
        if self.image is None:
            return

        # Preprocess the input image
        image_height, image_width, _ = self.image.shape
        blob = cv2.dnn.blobFromImage(self.image, 1/255, (416, 416), swapRB=True, crop=False)
        model.setInput(blob)

                # Perform object detection
        layer_names = model.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
        outputs = model.forward(output_layers)

        # Postprocess the detection results
        class_ids = []
        confidences = []
        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    w = int(detection[2] * image_width)
                    h = int(detection[3] * image_height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Draw bounding boxes and labels on the result image
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(indices))
            for i in indices.flatten():
                class_id = class_ids[i]
                label = classes[class_id]
                confidence = confidences[i]
                color = colors[class_id]
                x, y, w, h = boxes[i]
                cv2.rectangle(self.result_image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(self.result_image, f'{label}: {confidence:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                self.progress_bar.setValue(self.progress_bar.value() + 1)
            self.progress_bar.setVisible(False)

        # Display the result image
        self.display_image(self.result_image)

    def display_image(self, image):
        # Convert the OpenCV image to a Qt pixmap and display it in the image view widget
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(q_image)
        pixmap = pixmap.scaledToHeight(480)
        self.image_view.setPixmap(pixmap)

# Run the app
if __name__ == '__main__':
    app = QApplication([])
    window = App()
    app.exec_()


