import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tkinter as tk
from tkinter import ttk
from model_utils import load_models, load_feature_extractor, predict_text, predict_nb_text
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

device = torch.device("cpu")

# Defer heavy loading until first use so importing this module doesn't block
feature_extractor = None
model_entropy = None
model_kl = None
model_nb = None
model_emb = None
model_comb = None

# Store latest predictions for graphing
latest_predictions = None
graph_canvas = None
graph_figure = None

# Helper to show model availability text
def model_status_text(model):
    return "Loaded" if model is not None else "Missing"

def send_message():
    global feature_extractor, model_entropy, model_kl, model_nb, model_emb, model_comb

    # Lazy-load feature extractor and models on first message
    if feature_extractor is None:
        feature_extractor = load_feature_extractor()
        model_entropy, model_kl, model_emb, model_comb, model_nb = load_models()
        # update status labels
        lbl_status_entropy.config(text=f"Entropy Model:  {model_status_text(model_entropy)}")
        lbl_status_kl.config(text=f"KL Divergence Model: {model_status_text(model_kl)}")
        lbl_status_nb.config(text=f"NaiveBayes Model:    {model_status_text(model_nb)}")
        lbl_status_emb.config(text=f"Embedding Model:    {model_status_text(model_emb)}")
        lbl_status_comb.config(text=f"Combined Model:    {model_status_text(model_comb)}")
    user_text = entry.get().strip()
    if not user_text:
        return

    entry.config(state="disabled")
    send_button.config(state="disabled")
    try:
        chat_box.config(state="normal")
        chat_box.insert(tk.END, f"Input: {user_text}\n", "you")
        chat_box.see(tk.END)
        chat_box.config(state="disabled")

        # Get per-model predictions
        res_e = predict_text(user_text, feature_extractor, model_entropy, mode="entropy")
        res_kl = predict_text(user_text, feature_extractor, model_kl, mode="kl")
        res_nb = predict_nb_text(user_text, model_nb)
        res_emb = predict_text(user_text, feature_extractor, model_emb, mode="emb")
        res_comb = predict_text(user_text, feature_extractor, model_comb, mode="comb")

        # Store predictions for graphing
        global latest_predictions
        latest_predictions = {
            'Entropy': res_e.get('prob', 0),
            'KL': res_kl.get('prob', 0),
            'Naive Bayes': res_nb.get('prob', 0),
            'Embeddings': res_emb.get('prob', 0),
            'Combined': res_comb.get('prob', 0)
        }
        
        # Update results labels
        def fmt(r): return f"{r.get('label','?')} ({r.get('prob',0):.3f})"
        lbl_entropy_val.config(text=fmt(res_e))
        lbl_kl_val.config(text=fmt(res_kl))
        lbl_nb_val.config(text=fmt(res_nb))
        lbl_emb_val.config(text=fmt(res_emb))
        lbl_comb_val.config(text=fmt(res_comb))
        
        # Auto-display graph
        show_graph()

        # Append short summary to chat box with colored tag
        for name, r in [("Entropy", res_e), ("KL", res_kl), ("Emb", res_emb), ("Comb", res_comb), ("NB", res_nb)]:
            tag = "malicious" if r.get("label","Benign") == "Malicious" else "benign"
            chat_box.config(state="normal")
            chat_box.insert(tk.END, f"{name}: {r.get('label','?')} (p={r.get('prob',0):.3f})\n", tag)
            chat_box.see(tk.END)
            chat_box.config(state="disabled")
    finally:
        entry.config(state="normal")
        send_button.config(state="normal")
        entry.delete(0, tk.END)
        entry.focus_set()

def clear_chat():
    chat_box.config(state="normal")
    chat_box.delete("1.0", tk.END)
    chat_box.config(state="disabled")

def show_graph():
    global graph_figure, graph_canvas
    if latest_predictions is None:
        return
    
    models = list(latest_predictions.keys())
    probs = list(latest_predictions.values())
    
    if graph_figure is not None:
        plt.close(graph_figure)
    
    graph_figure = Figure(figsize=(5, 4), dpi=80)
    ax = graph_figure.add_subplot(111)
    
    colors = ['#5975A4', '#5F9E6E', '#B55D60', '#8C7AA2', '#A8860B']
    bars = ax.bar(models, probs, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
    
    ax.set_facecolor('#E8E8E8')
    graph_figure.patch.set_facecolor('#E8E8E8')
    
    ax.set_ylabel('Malicious Probability', fontsize=10)
    ax.set_title('Model Predictions', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='white')
    ax.set_axisbelow(True)
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        text_color = '#C41E3A' if prob > 0.5 else '#0A7A1A'
        ax.text(bar.get_x() + bar.get_width()/2, height / 2, f'{prob:.2f}', 
                ha='center', va='center', fontsize=9, color=text_color)
    
    ax.tick_params(axis='x', labelsize=8)
    graph_figure.tight_layout()
    
    if graph_canvas is not None:
        graph_canvas.get_tk_widget().destroy()
    
    graph_canvas = FigureCanvasTkAgg(graph_figure, master=graph_frame)
    graph_canvas.draw()
    graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# GUI setup
root = tk.Tk()
root.title("Prompt Injection Detector")
root.geometry("900x650")

style = ttk.Style(root)
style.theme_use("clam")

main = ttk.Frame(root, padding=8)
main.pack(fill=tk.BOTH, expand=True)

left = ttk.Frame(main)
left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,8))

right = ttk.Frame(main, width=240)
right.pack(side=tk.RIGHT, fill=tk.Y)

# Chat box
chat_box = tk.Text(left, height=20, width=60, wrap="word", state="disabled", font=("Consolas", 10))
chat_box.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
chat_box.tag_configure("you", foreground="#0B5FFF", font=("Consolas", 10, "bold"))
chat_box.tag_configure("malicious", foreground="#C41E3A")
chat_box.tag_configure("benign", foreground="#0A7A1A")

scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=chat_box.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_box.config(yscrollcommand=scrollbar.set)

# Results panel
ttk.Label(right, text="Model status", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,6))
lbl_status_entropy = ttk.Label(right, text=f"Entropy: {model_status_text(model_entropy)}")
lbl_status_entropy.pack(anchor="w")
lbl_status_kl = ttk.Label(right, text=f"KL:      {model_status_text(model_kl)}")
lbl_status_kl.pack(anchor="w")
lbl_status_nb = ttk.Label(right, text=f"NB:      {model_status_text(model_nb)}")
lbl_status_nb.pack(anchor="w")
lbl_status_emb = ttk.Label(right, text=f"Emb:     {model_status_text(model_emb)}")
lbl_status_emb.pack(anchor="w")
lbl_status_comb = ttk.Label(right, text=f"Comb:    {model_status_text(model_comb)}")
lbl_status_comb.pack(anchor="w")

ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)

ttk.Label(right, text="Latest predictions", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0,6))
lbl_entropy_val = ttk.Label(right, text="—")
lbl_entropy_val.pack(anchor="w")
lbl_kl_val = ttk.Label(right, text="—")
lbl_kl_val.pack(anchor="w")
lbl_nb_val = ttk.Label(right, text="—")
lbl_nb_val.pack(anchor="w")
lbl_emb_val = ttk.Label(right, text="—")
lbl_emb_val.pack(anchor="w")
lbl_comb_val = ttk.Label(right, text="—")
lbl_comb_val.pack(anchor="w")

# Graph frame for pie chart
graph_frame = ttk.Frame(right)
graph_frame.pack(fill=tk.BOTH, expand=True, pady=(8,0))

# Entry + buttons
bottom = ttk.Frame(root, padding=8)
bottom.pack(fill=tk.X)

entry = ttk.Entry(bottom, width=80)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,8))
entry.focus_set()

send_button = ttk.Button(bottom, text="Send", command=send_message)
send_button.pack(side=tk.LEFT)

clear_button = ttk.Button(bottom, text="Clear", command=clear_chat)
clear_button.pack(side=tk.LEFT, padx=(6,0))

# Bind Enter to the entry field so the chat stays responsive for multiple submissions
entry.bind("<Return>", lambda e: send_message())
entry.bind("<KP_Enter>", lambda e: send_message())

root.mainloop()