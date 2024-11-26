import cv2
import os
import psutil
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
from tkinter import messagebox
import subprocess
import sys
from ultralytics import YOLO
import yaml
import shutil
import random
from Xlib import display, X
import os.path

# Set the appearance mode and color theme for the GUI
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.title("Program Lock with Face Recognition")
        self.geometry("800x600")

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create sidebar frame
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        # Add logo label to sidebar
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Main Menu", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Add buttons to sidebar
        self.lock_button = ctk.CTkButton(self.sidebar_frame, text="Lock Programs", command=self.lock_all_programs)
        self.lock_button.grid(row=1, column=0, padx=20, pady=10)

        self.unlock_button = ctk.CTkButton(self.sidebar_frame, text="Unlock Programs", command=self.unlock_all_programs)
        self.unlock_button.grid(row=2, column=0, padx=20, pady=10)

        self.create_dataset_button = ctk.CTkButton(self.sidebar_frame, text="Create Dataset", command=self.create_dataset)
        self.create_dataset_button.grid(row=3, column=0, padx=20, pady=10)

        self.train_model_button = ctk.CTkButton(self.sidebar_frame, text="Train Model", command=self.train_model)
        self.train_model_button.grid(row=4, column=0, padx=20, pady=10)

        # Add appearance mode options
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                             command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))

        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Create scrollable frame for app list
        self.scrollable_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        # Initialize variables
        self.app_buttons = []
        self.locked_apps = []
        self.is_locked = False
        self.program_states = {}

        # Load desktop apps
        self.load_desktop_apps()

        # Start monitoring thread
        threading.Thread(target=self.monitor_locked_programs, daemon=True).start()

        # Initialize face recognition models
        self.face_model = YOLO('yolov8n-face.pt')
        self.trained_model = None
        self.dataset_path = "dataset"
        self.data_yaml_path = "data.yaml"

    def load_desktop_apps(self):
        """Load and display desktop apps from the applications directory"""
        apps_dir = "/usr/share/applications"
        
        for filename in os.listdir(apps_dir):
            if filename.endswith('.desktop'):
                app_path = os.path.join(apps_dir, filename)
                app_name = os.path.splitext(filename)[0]
                
                app_frame = ctk.CTkFrame(self.scrollable_frame)
                app_frame.grid(sticky="ew", padx=5, pady=5)
                app_frame.grid_columnconfigure(1, weight=1)

                app_button = ctk.CTkCheckBox(app_frame, text=app_name, 
                                           command=lambda path=app_path: self.toggle_app_lock(path))
                app_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")

                self.app_buttons.append((app_button, app_path))

    def toggle_app_lock(self, app_path):
        """Toggle lock status for an app"""
        if app_path in self.locked_apps:
            self.locked_apps.remove(app_path)
        else:
            self.locked_apps.append(app_path)

    def lock_all_programs(self):
        """Lock all selected programs"""
        self.is_locked = True
        messagebox.showinfo("Notification", "Selected programs have been locked.")

    def unlock_all_programs(self):
        """Attempt to unlock all programs using face recognition"""
        if self.face_recognition():
            self.is_locked = False
            self.program_states.clear()
            messagebox.showinfo("Notification", "Face recognized. Programs unlocked.")
        else:
            messagebox.showwarning("Warning", "Face not recognized. Programs remain locked.")

    def is_program_running(self, program_name):
        """Check if a program is currently running"""
        for proc in psutil.process_iter(['name']):
            if proc.info['name'].lower() == program_name.lower():
                return True
        return False

    def lock_screen(self):
        """Lock the screen using gnome-screensaver"""
        try:
            subprocess.run(['gnome-screensaver-command', '-l'])
        except:
            try:
                # Alternative method using dbus
                subprocess.run(['dbus-send', '--type=method_call', '--dest=org.gnome.ScreenSaver', 
                              '/org/gnome/ScreenSaver', 'org.gnome.ScreenSaver.Lock'])
            except:
                messagebox.showwarning("Warning", "Could not lock screen. Please install gnome-screensaver.")

    def face_recognition(self):
        """Perform face recognition"""
        if self.trained_model is None:
            try:
                self.trained_model = YOLO('trained_face_model.pt')
            except:
                messagebox.showwarning("Warning", "Please train the model first.")
                return False

        vid_cam = cv2.VideoCapture(0)
        start_time = time.time()
        while time.time() - start_time < 10:  # 10 seconds for face recognition
            _, image_frame = vid_cam.read()
            
            results = self.trained_model(image_frame)
            
            if len(results[0].boxes) > 0:
                vid_cam.release()
                cv2.destroyAllWindows()
                return True
            
            cv2.imshow('Face Recognition', image_frame)
            
            if cv2.waitKey(100) & 0xFF == 27:
                break
        
        vid_cam.release()
        cv2.destroyAllWindows()
        return False

    def monitor_locked_programs(self):
        """Monitor and control locked programs"""
        while True:
            if self.is_locked:
                for app_path in self.locked_apps:
                    program_name = os.path.basename(app_path)
                    
                    is_running = self.is_program_running(program_name)
                    
                    if is_running and not self.program_states.get(program_name, False):
                        subprocess.run(['killall', program_name])
                        
                        if self.face_recognition():
                            subprocess.Popen(['gtk-launch', program_name])
                            self.program_states[program_name] = True
                        else:
                            messagebox.showwarning("Warning", "Face not recognized. Program will not run.")
                            self.program_states[program_name] = False
                    
                    elif not is_running:
                        self.program_states[program_name] = False
            
            time.sleep(1)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        """Change the appearance mode of the GUI"""
        ctk.set_appearance_mode(new_appearance_mode)

    def train_model(self):
        """Train the face recognition model"""
        messagebox.showinfo("Model Training", "Model training started. This process may take several minutes.")

        if not os.path.exists(self.dataset_path):
            messagebox.showerror("Error", "Dataset not found. Please create the dataset first.")
            return

        # Prepare data.yaml file
        with open(self.data_yaml_path, 'w') as file:
            yaml.dump({
                'train': self.dataset_path,
                'val': self.dataset_path,
                'nc': 1,  # number of classes
                'names': ['face']  # class names
                        }, file)

        # Create and train the model
        try:
            # Load a pretrained YOLO model
            self.trained_model = YOLO('yolov8n.pt')

            # Train the model
            results = self.trained_model.train(
                data=self.data_yaml_path,
                epochs=50,
                imgsz=640,
                batch=16,
                device='cpu'  # Use 'cuda' if you have a compatible GPU
                )

            # Save the trained model
            self.trained_model.save('trained_face_model.pt')

            messagebox.showinfo("Model Training", "Model training completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error in model training: {str(e)}")

    def create_dataset(self):
        """Create a dataset for face recognition"""
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        vid_cam = cv2.VideoCapture(0)
        face_id = 1
        count = 0

        while True:
            _, image_frame = vid_cam.read()
        
            results = self.face_model(image_frame)
        
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(image_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                    count += 1
                
                    face = image_frame[y1:y2, x1:x2]
                    cv2.imwrite(f"{self.dataset_path}/User.{face_id}.{count}.jpg", face)

                    cv2.imshow('Creating Dataset!!!', image_frame)

            if cv2.waitKey(100) & 0xFF == 27:
                break
            elif count > 100:
                break

        vid_cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Create Dataset", "Dataset created successfully.")

if __name__ == "__main__":
    app = App()
    app.mainloop()