import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # Import PIL modules for image handling
import plotly.express as px
from preprocessing.data_scraper import DataScraper
from preprocessing.preprocessing import Preprocessing
from preprocessing.featureExtraction import WordEmbeddings_Handler
from preprocessing.dimensionality_reduction import PCA_Handler
from model.kmeans_model import KMeans_Model

# Global variables to store the results
processed_X = None
X_embeddings = None
X_pca = None
model = None
urls = None

# Splash screen setup
splash = tk.Tk()
splash.title("Loading...")
splash.geometry("400x400")  # Adjusted geometry to fit both image and progress bar

# Load and display the image
image_path = r"C:\Users\fuzzi\Downloads\MidProject\MidProject1\src\ai.jpg"
if os.path.exists(image_path):
    img = Image.open(image_path)
    img = img.resize((400, 300), Image.LANCZOS)  # Resize image as needed

    # Create a Tkinter label and display the image
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(splash, image=photo)
    label.pack()

# Progress bar setup
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(splash, variable=progress_var, maximum=100)
progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

# Function to update progress bar
def update_progress(value):
    progress_var.set(value)
    splash.update_idletasks()

# Function to start the main window after loading
def start_main_window():
    splash.destroy()
    global window
    window = tk.Tk()
    window.title("DB Organizer")

    welcome_message = "Welcome to Database Organizer!\n\nPlease send the URL for the item you want to categorize or type 'done' to see the results."

    global chat_display
    chat_display = tk.Text(window, width=60, height=20)
    chat_display.insert(tk.END, welcome_message + "\n\n")
    chat_display.pack()

    global entry
    entry = tk.Entry(window, width=50)
    entry.pack(pady=10)

    send_button = tk.Button(window, text="Send", command=lambda: process_message(chat_display, entry))
    send_button.pack()

    window.mainloop()

# Function to process user messages
def process_message(chat_display, entry):
    global processed_X, X_embeddings, X_pca, model, urls
    message = entry.get()
    
    if message.lower() == "done":
        if model and X_pca is not None:
            chat_display.insert(tk.END, "Cluster 1: Cultural and Artistic Heritage:\n")
            for i, label in enumerate(model.labels_silhouette):
                if label == 0:
                    chat_display.insert(tk.END, f"{urls[i]}\n")
            
            chat_display.insert(tk.END, "\nCluster 2: Symbols of Peace and Historical Commemoration\n")
            for i, label in enumerate(model.labels_silhouette):
                if label == 1:
                    chat_display.insert(tk.END, f"{urls[i]}\n")

            display_graphs()
        else:
            chat_display.insert(tk.END, "No processed data found. Please provide URLs first.\n\n")
    elif message.startswith("https://"):
        url = message  
        scraper = DataScraper(url)
        scraper.fetch_content()
        scraper.parse_content()
        scraper.retrieve_data()
        text = scraper.text 
        chat_display.insert(tk.END, "Data retrieved from URL. Now you can proceed.\n\n")
    else:
        chat_display.insert(tk.END, f"You: {message}\n\n")
    
    entry.delete(0, tk.END)

# Function to display graphs
def display_graphs():
    fig1 = r"visualizations\initial_plot.png"
    fig2 = r"visualizations\final_plot_silhouette.png"
    fig3 = r"visualizations\final_plot_elbow.png"

    if os.path.exists(fig1):
        os.system(f"start {fig1}")
    if os.path.exists(fig2):
        os.system(f"start {fig2}")
    if os.path.exists(fig3):
        os.system(f"start {fig3}")

def label_data(): 
    return

# Function to fit model and update progress bar
def fit_model(progress_callback):
    global processed_X, X_embeddings, X_pca, model, urls
    gifts_df = pd.read_csv(r"src\list_gifts_un_complete.csv")
    urls = gifts_df['URL']
    
    X = []
    for i, url in enumerate(urls):
        scraper = DataScraper(url)
        scraper.fetch_content()
        scraper.parse_content()
        scraper.retrieve_data()
        data = scraper.text
        X.append(data)
        progress_callback(int((i / len(urls)) * 20))
    
    processed_X = []
    for i, text in enumerate(X):
        Preprocess = Preprocessing()
        tokens = Preprocess.preprocess(text)
        processed_X.append(tokens)
        progress_callback(20 + int((i / len(X)) * 20))
        
    TextEmbedder = WordEmbeddings_Handler()
    TextEmbedder.build_model(processed_X)
    TextEmbedder.train_model(processed_X)
    X_embeddings = TextEmbedder.infer_matrix(processed_X)
    progress_callback(60)
    
    pca = PCA_Handler(n_components=2)
    X_pca = pca.fit_transform(X_embeddings)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1])
    # path1 = ".."
    # path2 = "visualizations"
    # path3 = "initial_plot.png"
    # path = os.path.join(path1, path2, path3)
    path = r"visualizations\initial_plot.png"
    fig.write_image(path)
    progress_callback(80)
    
    model = KMeans_Model()
    model.fit_silhouette_analysis(X_pca)
    model.fit_elbow_analysis(X_pca)

    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=model.labels_silhouette)
    # path1 = ".."
    # path2 = "visualizations"
    # path3 = "final_plot_silhouette.png"
    # path = os.path.join(path1, path2, path3)
    path = r"visualizations\final_plot_silhouette.png"
    fig.write_image(path)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=model.labels_elbow)
    # path1 = ".."
    # path2 = "visualizations"
    # path3 = "final_plot_elbow.png"
    # path = os.path.join(path1, path2, path3)
    path = r"visualizations\final_plot_elbow.png"
    fig.write_image(path)

    progress_callback(100)

# Run the fit_model function and update the progress bar
splash.after(100, lambda: fit_model(update_progress))
splash.after(1000, start_main_window)
splash.mainloop()
