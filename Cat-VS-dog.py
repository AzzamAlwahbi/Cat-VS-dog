import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageOps, ImageTk
import numpy as np
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# GUI Application
class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("🐱🐶 Cat vs Dog Classifier")
        master.geometry("500x600")
        master.resizable(False, False)
        master.configure(bg="#f0f0f0")

        # Title Label
        self.title_label = Label(
            master,
            text="Cat vs Dog Image Classifier",
            font=("Arial", 20, "bold"),
            bg="#4a7abc",
            fg="white",
            pady=10
        )
        self.title_label.pack(fill="x")

        # Instructions
        self.label = Label(
            master,
            text="Upload an image to classify",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.label.pack(pady=15)

        # Upload Button
        self.upload_button = Button(
            master,
            text="Upload Image",
            font=("Arial", 12, "bold"),
            bg="#4a7abc",
            fg="white",
            width=20,
            command=self.upload_image
        )
        self.upload_button.pack(pady=5)

        # Image Frame with border
        self.image_frame = Frame(master, bd=2, relief="solid", bg="white")
        self.image_frame.pack(pady=15)
        self.image_label = Label(self.image_frame, bg="white")
        self.image_label.pack()

        # Result Label
        self.result_label = Label(
            master,
            text="",
            font=("Arial", 14),
            bg="#f0f0f0",
            fg="#333333",
            justify="center"
        )
        self.result_label.pack(pady=15)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            # Load and display the image
            pil_image = Image.open(file_path).convert("RGB")
            display_image = pil_image.resize((300, 300))
            tk_image = ImageTk.PhotoImage(display_image)
            self.image_label.configure(image=tk_image)
            self.image_label.image = tk_image

            # Classify the image
            prediction, confidence = self.classify_image(pil_image)

            # Show the result
            result_text = f"Prediction: {prediction}\nConfidence: {confidence*100:.2f}%"
            self.result_label.config(text=result_text)

    def classify_image(self, pil_image):
        # Prepare the image
        size = (224, 224)
        image = ImageOps.fit(pil_image, size, Image.Resampling.LANCZOS)

        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predict
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()[2:]  # Remove index and whitespace
        confidence_score = prediction[0][index]

        return class_name, confidence_score

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
