# Real-time Garbage Collector using ResNet152

This project is a real-time garbage classification system that uses a deep learning model to classify waste and a servo motor to control a trash bin lid. The system is built with Python, TensorFlow/Keras, and OpenCV, and it interacts with an Arduino to control the hardware.

## Features

*   **Real-time Garbage Classification:** The system uses a webcam to capture live video and classifies garbage in real-time.
*   **Hardware Integration:** It controls a servo motor connected to an Arduino to automatically open and close the trash bin lid.
*   **Voice Feedback:** The system provides voice feedback using Google Text-to-Speech (gTTS) to announce the detected object and the status of the trash bin.
*   **Deep Learning Model:** The project uses a pre-trained ResNet152 model for image classification, fine-tuned for garbage classification.

## Requirements

### Software

*   Python 3.7+
*   TensorFlow
*   Keras
*   OpenCV-python
*   pyfirmata
*   gTTS
*   playsound

### Hardware

*   Webcam
*   Arduino Uno
*   Servo Motor
*   Jumper Wires

## Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Real-time-garbage-collector-using-resnet152.git
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file)*

3.  **Set up the hardware:**
    *   Connect the servo motor to the Arduino.
    *   Connect the Arduino to your computer.
    *   Make sure the Arduino is on the correct port (e.g., `COM3`). You might need to change the port in `controller.py`.

4.  **Run the application:**
    ```bash
    python test.py
    ```

## How it works

1.  The `test.py` script captures video from the webcam.
2.  Each frame is preprocessed and fed into the pre-trained garbage classification model (`model.h5`).
3.  If the model detects an object with a high probability, it identifies the object.
4.  If the same object is detected for a certain number of consecutive frames, the script sends a signal to the Arduino to open the trash bin lid.
5.  After a short delay, the script sends another signal to close the lid.
6.  The `controller.py` script handles the communication with the Arduino and controls the servo motor.

## Future Scope

*   **Expand the dataset:** The model can be trained on a larger dataset with more classes of garbage to improve its accuracy and robustness.
*   **Improve the hardware:** A more robust and durable hardware setup can be built for the trash bin.
*   **Add more features:** The system can be extended to include features like sorting the garbage into different compartments.
