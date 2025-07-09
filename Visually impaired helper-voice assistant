import cv2
import speech_recognition as sr
import pyttsx3
from ultralytics import YOLO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Function to speak the direction
def speak_direction(direction):
    engine.say(direction)
    engine.runAndWait()

def give_direction(x_center, width):
    if x_center < width // 3:
        direction = "Move right"
        print(direction)
        speak_direction(direction)
    elif x_center > 2 * width // 3:
        direction = "Move left"
        print(direction)
        speak_direction(direction)
    else:
        direction = "STOP Object Ahead"
        print(direction)
        speak_direction(direction)

# Function to start real-time object detection
def path_finder():
    # Open video stream
    cap = cv2.VideoCapture(0)

    # Load YOLO model
    model = YOLO("yolov8n.pt")
    model.info()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions
        height, width, _ = frame.shape

        # Run YOLO model on the current frame
        results = model(frame)

        # Process detections
        for result in results:
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = (x1 + x2) / 2

                # Give direction based on object's center position
                give_direction(x_center, width)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream
    cap.release()

# Function to start BLIP image captioning with interruption
def blip_image_captioning():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Starting BLIP Image Captioning. Say 'stop BLIP' to return to the main menu.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL Image format
        raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Provide a prompt for the model
        text = "a photography of"  # Optional context

        # Prepare the inputs for the model
        inputs = processor(raw_image, text, return_tensors="pt")

        # Generate the caption using the model
        out = model.generate(**inputs)

        # Decode the caption
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Speak the generated caption
        engine.say(caption)
        engine.runAndWait()

        # Display the frame with the caption
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, caption, (20, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame in a window
        cv2.imshow('BLIP Image Captioning', frame)

        # Check for stop command while showing video
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                command = recognizer.recognize_google(audio).lower()
                if "stop blip" in command:
                    print("Stopping BLIP and returning to the main menu...")
                    speak_direction("Stopping BLIP and returning to the main menu.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            except sr.UnknownValueError:
                pass  # Ignore unrecognized speech
            except sr.RequestError:
                print("Speech recognition service error.")
            except Exception as e:
                print(f"Error: {e}")

        # Check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to listen for voice commands with retry mechanism
def listen_for_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Listening for command... Say 'open' to start YOLO, 'detect the objects around me' for BLIP, 'close' to stop BLIP, or 'quit' to exit.")

    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")
                if "open" in command:
                    print("Opening YOLO object detection...")
                    speak_direction("Opening the object detection.")
                    path_finder()
                elif "detect the objects around me" in command:
                    print("Starting BLIP image captioning...")
                    speak_direction("Starting image captioning to detect objects around you.")
                    blip_image_captioning()
                elif "close" in command:
                    print("Closing BLIP or active process...")
                    speak_direction("Closing active process. Returning to listening mode.")
                    cv2.destroyAllWindows()
                elif "quit" in command:
                    print("Quitting the application...")
                    speak_direction("Quitting the application. Goodbye.")
                    break  # Exit the entire program
                else:
                    print("Command not recognized. Try again.")
                    speak_direction("Command not recognized. Try again.")
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
                speak_direction("Sorry, I could not understand the audio. Please try again.")
            except sr.RequestError:
                print("Could not request results from the speech recognition service.")
                speak_direction("Could not request results from the speech recognition service. Please try again.")
            except Exception as e:
                print(f"Error: {e}")
                speak_direction(f"An error occurred: {e}")
            print("Listening again...")

if _name_ == "_main_":
    listen_for_command()
