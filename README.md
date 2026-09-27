# Computer-Vision-Project
# 🔍 AI Object Detection using YOLO & OpenCV

An AI-based **real-time object detection application** built with **Python, YOLO, OpenCV, and Streamlit**. The application can detect objects from a **webcam, image, or video file** and display the detected objects with bounding boxes and confidence scores.

## 🚀 Features

* 🎥 **Webcam Detection** – Detect objects in real time using your webcam.
* 🖼️ **Image Detection** – Upload an image and detect objects in it.
* 🎬 **Video Detection** – Upload a video and perform object detection frame by frame.
* 📦 **YOLO Model** – Uses YOLO for fast and accurate object detection.
* 📊 **Confidence Control** – Adjust the detection confidence threshold.
* 🏷️ **Bounding Boxes & Labels** – Displays detected objects with their names and confidence scores.
* 🌐 **Streamlit Interface** – Simple and interactive web-based interface.
* ⚡ **Real-Time Processing** – Optimized for fast detection.

## 🛠️ Technologies Used

| Technology | Purpose                        |
| ---------- | ------------------------------ |
| Python     | Core programming language      |
| YOLO       | Object detection model         |
| OpenCV     | Image/video processing         |
| Streamlit  | Web-based user interface       |
| NumPy      | Numerical and array operations |

## 📁 Project Structure

```text
AI-Object-Detection/
│
├── app.py                  # Streamlit application
├── detector.py             # Object detection logic
├── yolo11n.pt              # YOLO model
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
│
└── screenshots/            # Project screenshots
    ├── webcam.png
    ├── image.png
    └── video.png
```

> File names can be changed according to the actual files in your repository.

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mai-khush-hu/AI-Object-Detection.git
```

```bash
cd AI-Object-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file yet:

```bash
pip install ultralytics opencv-python streamlit numpy
```

## ▶️ Run the Project

Start the Streamlit application using:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

Open the URL in your browser.

## 🎯 How to Use

### 1. Webcam Mode

Select **Webcam** from the application.

Allow camera access when prompted.

The application will detect objects in real time and display:

* Object name
* Bounding box
* Confidence score

### 2. Image Mode

Select **Image** and upload an image.

The model will process the image and display the detected objects.

### 3. Video Mode

Select **Video** and upload a video file.

The application processes the video frame by frame and detects objects throughout the video.

## 🧠 Object Detection Pipeline

```text
Input
  │
  ├── Webcam
  ├── Image
  └── Video
       │
       ▼
   OpenCV
       │
       ▼
   YOLO Model
       │
       ▼
 Object Detection
       │
       ▼
Bounding Boxes
 + Labels
 + Confidence
       │
       ▼
 Streamlit Interface
```

## 📸 Screenshots

### 🎥 Webcam Detection

Add your screenshot here:

```markdown
![Webcam Detection](screenshots/webcam.png)
```

### 🖼️ Image Detection

```markdown
![Image Detection](screenshots/image.png)
```

### 🎬 Video Detection

```markdown
![Video Detection](screenshots/video.png)
```

## 📊 Model

This project uses a YOLO model for object detection.

The model identifies objects and returns:

```text
Object Class
Confidence Score
Bounding Box Coordinates
```

For example:

```text
Person       0.92
Car          0.87
Bottle       0.81
Laptop       0.76
```

## 👨‍💻 Author

**Khushal Kumar**

B.Tech Computer Science & Engineering
Artificial Intelligence & Machine Learning

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub!

---

### 📜 License

This project is created for educational and learning purposes.
