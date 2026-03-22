import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tkinter as tk
from tkinter import ttk
from model_utils import load_models, load_feature_extractor, predict_text
import torch

device = torch.device("cpu")

# Defer heavy loading until first use so importing this module doesn't block
feature_extractor = None
model_entropy = None
model_kl = None
model_emb = None
model_comb = None

# Helper to show model availability text
def model_status_text(model):
    return "Loaded" if model is not None else "Missing"

def send_message():
    global feature_extractor, model_entropy, model_kl, model_emb, model_comb

    # Lazy-load feature extractor and models on first message
    if feature_extractor is None:
        feature_extractor = load_feature_extractor()
        model_entropy, model_kl, model_emb, model_comb = load_models()
        # update status labels
        lbl_status_entropy.config(text=f"Entropy: {model_status_text(model_entropy)}")
        lbl_status_kl.config(text=f"KL:      {model_status_text(model_kl)}")
        lbl_status_emb.config(text=f"Emb:     {model_status_text(model_emb)}")
        lbl_status_comb.config(text=f"Comb:    {model_status_text(model_comb)}")
    user_text = entry.get().strip()
    if not user_text:
        return

    # disable while processing
    entry.config(state="disabled")
    send_button.config(state="disabled")

    chat_box.config(state="normal")
    chat_box.insert(tk.END, f"You: {user_text}\n", "you")
    chat_box.see(tk.END)
    chat_box.config(state="disabled")

    # Get per-model predictions
    res_e = predict_text(user_text, feature_extractor, model_entropy, mode="entropy")
    res_kl = predict_text(user_text, feature_extractor, model_kl, mode="kl")
    res_emb = predict_text(user_text, feature_extractor, model_emb, mode="emb")
    res_comb = predict_text(user_text, feature_extractor, model_comb, mode="comb")

    # Update results labels
    def fmt(r): return f"{r.get('label','?')} ({r.get('prob',0):.3f})"
    lbl_entropy_val.config(text=fmt(res_e))
    lbl_kl_val.config(text=fmt(res_kl))
    lbl_emb_val.config(text=fmt(res_emb))
    lbl_comb_val.config(text=fmt(res_comb))

    # Append short summary to chat box with colored tag
    for name, r in [("Entropy", res_e), ("KL", res_kl), ("Emb", res_emb), ("Comb", res_comb)]:
        tag = "malicious" if r.get("label","Benign") == "Malicious" else "benign"
        chat_box.config(state="normal")
        chat_box.insert(tk.END, f"{name}: {r.get('label','?')} (p={r.get('prob',0):.3f})\n", tag)
        chat_box.see(tk.END)
        chat_box.config(state="disabled")

    entry.delete(0, tk.END)
    entry.config(state="normal")
    send_button.config(state="normal")
    entry.focus_set()

def clear_chat():
    chat_box.config(state="normal")
    chat_box.delete("1.0", tk.END)
    chat_box.config(state="disabled")

# GUI setup
root = tk.Tk()
root.title("Prompt Injection Detector")
root.geometry("800x420")

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
lbl_emb_val = ttk.Label(right, text="—")
lbl_emb_val.pack(anchor="w")
lbl_comb_val = ttk.Label(right, text="—")
lbl_comb_val.pack(anchor="w")

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

# Bind Enter
root.bind("<Return>", lambda e: send_message())

root.mainloop()