
import pyttsx3

def speak_letter(letter):
    engine = pyttsx3.init()
    engine.say(f"The letter is {letter}")
    engine.runAndWait()

speak_letter(predicted_label)
# predicted word
engine.setProperty('volume', 0.9)