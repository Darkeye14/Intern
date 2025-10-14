import speech_recognition as sr
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import pyaudio
import wave
import os
from datetime import datetime

class SpeechToTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech to Text Converter")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Initialize recognizer
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        
        # Create GUI
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text display
        self.text_display = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=60, height=20)
        self.text_display.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        self.record_btn = ttk.Button(
            button_frame, 
            text="🎤 Start Listening", 
            command=self.toggle_listening,
            style="Accent.TButton"
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(
            button_frame,
            text="🗑️ Clear Text",
            command=self.clear_text
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(
            button_frame,
            text="💾 Save to File",
            command=self.save_to_file
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('Accent.TButton', font=('Helvetica', 10, 'bold'))
        
    def toggle_listening(self):
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()
    
    def start_listening(self):
        self.is_listening = True
        self.record_btn.config(text="⏹️ Stop Listening")
        self.status_var.set("Listening... Speak now!")
        
        # Start listening in a separate thread
        self.listening_thread = threading.Thread(target=self.listen_audio, daemon=True)
        self.listening_thread.start()
    
    def stop_listening(self):
        self.is_listening = False
        self.record_btn.config(text="🎤 Start Listening")
        self.status_var.set("Ready")
    
    def listen_audio(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            self.status_var.set("Adjusting for ambient noise...")
            self.root.update()
            
            while self.is_listening:
                try:
                    self.status_var.set("Listening... Speak now!")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Recognize speech using Google Web Speech API
                    try:
                        text = self.recognizer.recognize_google(audio)
                        self.text_display.insert(tk.END, text + "\n")
                        self.text_display.see(tk.END)
                    except sr.UnknownValueError:
                        self.status_var.set("Could not understand audio")
                    except sr.RequestError as e:
                        self.status_var.set(f"Error: {e}")
                    
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    self.status_var.set(f"Error: {str(e)}")
                    self.stop_listening()
    
    def clear_text(self):
        self.text_display.delete(1.0, tk.END)
        self.status_var.set("Text cleared")
    
    def save_to_file(self):
        text = self.text_display.get(1.0, tk.END).strip()
        if not text:
            self.status_var.set("No text to save")
            return
            
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speech_to_text_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
            self.status_var.set(f"Text saved to {filename}")
        except Exception as e:
            self.status_var.set(f"Error saving file: {str(e)}")

def main():
    root = tk.Tk()
    app = SpeechToTextApp(root)
    
    # Set application icon and style
    try:
        root.iconbitmap('microphone.ico')  # You can add an icon file if desired
    except:
        pass  # Continue without icon if not found
    
    # Center the window
    window_width = 600
    window_height = 500
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()