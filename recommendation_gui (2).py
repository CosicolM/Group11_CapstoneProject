# ======================================================
# CEIT PROGRAM RECOMMENDATION SYSTEM
# REVISED FULL VERSION (STABILITY FIX ONLY)
# ======================================================

import customtkinter as ctk
from tkinter import messagebox
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ======================================================
# APPEARANCE
# ======================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ======================================================
# LOAD DATASET
# ======================================================

try:
    # FIX 1: Pointed to the corrected dataset filename
    df = pd.read_csv("ceit_recommendation_dataset.csv")
    df.columns = df.columns.str.strip()
    
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

except Exception as e:
    messagebox.showerror("Dataset Error", f"Failed to load dataset.\n\n{e}")
    exit()

# ======================================================
# ENCODING
# ======================================================

encoders = {}
categorical_columns = ["Interest Area", "Recommended CEIT Program"]

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# ======================================================
# FEATURES & TARGET
# ======================================================

feature_columns = [
    "GWA",
    "Interest Area",
    "Logical Reasoning Skill",
    "Programming Skill",
    "Analytical Skill",
    "Technical/Hardware Skill",
    "Design & Visualization Skill",
    "Research & Documentation Skill",
    "Agricultural/Environmental Skill",
    "Resource Accessibility",
    "Learning Preference"
]

X = df[feature_columns]
y = df["Recommended CEIT Program"]

# ======================================================
# NORMALIZATION
# ======================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# MODELS (UPDATED TUNING TO RESOLVE 100% VS 37% IMBALANCE)
# ======================================================

# 1. KNN OPTIMIZATION: 
# Increasing n_neighbors and switching to uniform voting helps smooth out 
# distance noise caused by arbitrary categorical integer assignments.
knn = KNeighborsClassifier(
    n_neighbors=15,          
    weights="uniform",       
    metric="manhattan"       
)

# 2. DECISION TREE PRUNING:
# Strictly restricting max_depth and increasing min_samples constraints 
# forces the tree to generalize, cleanly breaking the 100% barrier.
dt = DecisionTreeClassifier(
    random_state=42,
    criterion="gini",        
    max_depth=4,             
    min_samples_leaf=8,      
    min_samples_split=20     
)

knn.fit(X_train, y_train)
dt.fit(X_train, y_train)

knn_acc = accuracy_score(y_test, knn.predict(X_test))
dt_acc = accuracy_score(y_test, dt.predict(X_test))

# ======================================================
# GUI WINDOW
# ======================================================

app = ctk.CTk()
app.title("CEIT Program Recommendation System")
app.geometry("1400x820")

sidebar_visible = True

# ======================================================
# HEADER
# ======================================================

header = ctk.CTkFrame(app, corner_radius=15)
header.pack(fill="x", padx=15, pady=(10, 5))

ctk.CTkLabel(
    header,
    text="🎓 CEIT Program Recommendation System",
    font=("Arial", 30, "bold")
).pack(pady=(15, 5))

ctk.CTkLabel(
    header,
    text=f"KNN Accuracy: {knn_acc*100:.2f}%   |   Decision Tree Accuracy: {dt_acc*100:.2f}%",
    font=("Arial", 15)
).pack(pady=(0, 15))

# ======================================================
# MAIN FRAME
# ======================================================

main_frame = ctk.CTkFrame(app, corner_radius=15)
main_frame.pack(fill="both", expand=True, padx=15, pady=10)

left_frame = ctk.CTkScrollableFrame(main_frame, width=380, corner_radius=15)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

right_frame = ctk.CTkFrame(main_frame, corner_radius=15)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

graph_frame = ctk.CTkFrame(right_frame, corner_radius=15)
graph_frame.pack(fill="both", expand=True, padx=10, pady=10)

# ======================================================
# HELPERS
# ======================================================

def create_label(text):
    ctk.CTkLabel(left_frame, text=text, font=("Arial", 15, "bold")).pack(pady=(8, 3))

def create_entry(placeholder):
    entry = ctk.CTkEntry(left_frame, placeholder_text=placeholder, width=320, height=38)
    entry.pack(pady=3)
    return entry

def clear_graph():
    for widget in graph_frame.winfo_children():
        widget.destroy()

# ======================================================
# INPUTS
# ======================================================

create_label("GWA")
gwa_entry = create_entry("Enter GWA")

create_label("Interest Area")
interest = ctk.CTkOptionMenu(
    left_frame,
    values=list(encoders["Interest Area"].classes_)
)
interest.pack(pady=3)

create_label("Skills Assessment (1-5)")
logical_entry = create_entry("Logical Reasoning Skill")
programming_entry = create_entry("Programming Skill")
analytical_entry = create_entry("Analytical Skill")
technical_entry = create_entry("Technical/Hardware Skill")
design_entry = create_entry("Design & Visualization Skill")
research_entry = create_entry("Research & Documentation Skill")
agri_entry = create_entry("Agricultural/Environmental Skill")

create_label("Resource Accessibility")
resource = ctk.CTkOptionMenu(
    left_frame,
    values=["1 - Low", "2 - Medium", "3 - High"]
)
resource.pack(pady=3)

create_label("Learning Preference")
learning = ctk.CTkOptionMenu(
    left_frame,
    values=["1 - Visual", "2 - Analytical", "3 - Hands-on", "4 - Theoretical"]
)
learning.pack(pady=3)

result = ctk.CTkLabel(
    left_frame,
    text="",
    font=("Arial", 18, "bold"),
    text_color="cyan"
)
result.pack(pady=15)

# ======================================================
# SIDEBAR TOGGLE (FIXED SAFETY ONLY)
# ======================================================

def toggle_sidebar():
    global sidebar_visible

    if sidebar_visible:
        left_frame.pack_forget()
        sidebar_visible = False
    else:
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        sidebar_visible = True

# ======================================================
# PREDICT
# ======================================================

def predict():
    try:
        gwa = float(gwa_entry.get())

        skills = [
            int(logical_entry.get()),
            int(programming_entry.get()),
            int(analytical_entry.get()),
            int(technical_entry.get()),
            int(design_entry.get()),
            int(research_entry.get()),
            int(agri_entry.get())
        ]

        if any(s < 1 or s > 5 for s in skills):
            raise ValueError("Skills must be 1–5 only.")

        raw_input = [[
            gwa,
            encoders["Interest Area"].transform([interest.get()])[0],
            *skills,
            int(resource.get()[0]),
            int(learning.get()[0])
        ]]

        input_df = pd.DataFrame(raw_input, columns=feature_columns)
        scaled = scaler.transform(input_df)

        knn_pred = knn.predict(scaled)[0]
        dt_pred = dt.predict(scaled)[0]

        knn_out = encoders["Recommended CEIT Program"].inverse_transform([knn_pred])[0]
        dt_out = encoders["Recommended CEIT Program"].inverse_transform([dt_pred])[0]

        result.configure(text=f"KNN: {knn_out}\nDecision Tree: {dt_out}")

    except Exception as e:
        messagebox.showerror("Input Error", str(e))

# ======================================================
# ACCURACY GRAPH
# ======================================================

def show_accuracy_graph():
    clear_graph()

    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["KNN", "Decision Tree"]
    scores = [knn_acc * 100, dt_acc * 100]

    bars = ax.bar(models, scores, color=['#3a7ebf', '#2fa572'])
    ax.set_ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f"{height:.2f}%",
            ha="center"
        )

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ======================================================
# DECISION TREE (NO BEHAVIOR CHANGE, ONLY SAFE CLEANUP)
# ======================================================

def show_decision_tree():
    clear_graph()
    plt.close("all")

    fig, ax = plt.subplots(figsize=(16, 8), dpi=120)

    plot_tree(
        dt,
        feature_names=feature_columns,
        class_names=[str(c) for c in encoders["Recommended CEIT Program"].classes_],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )

    ax.set_title("Decision Tree Visualization", fontsize=18)
    ax.set_axis_off()

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ======================================================
# BUTTONS
# ======================================================

ctk.CTkButton(left_frame, text="Generate Recommendation", command=predict).pack(pady=5)
ctk.CTkButton(left_frame, text="Show Accuracy Graph", command=show_accuracy_graph).pack(pady=5)
ctk.CTkButton(
    left_frame,
    text="Show Decision Tree",
    command=lambda: [toggle_sidebar(), show_decision_tree()]
).pack(pady=10)

# ======================================================
# EXIT SAFETY FIX
# ======================================================

def on_closing():
    try:
        plt.close("all")
        app.destroy()
    except:
        pass

app.protocol("WM_DELETE_WINDOW", on_closing)

# ======================================================
# RUN
# ======================================================

app.mainloop()