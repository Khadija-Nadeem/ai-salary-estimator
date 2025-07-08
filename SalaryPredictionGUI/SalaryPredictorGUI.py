import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import joblib
import os

class SalaryPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Location', 'Skills']
        self.model_trained = False
        self.model_performance = {}
        
    def load_and_prepare_data(self, csv_file=None):
        try:
            if csv_file is None:
                csv_file = filedialog.askopenfilename(
                    title="Select Salary Data CSV File",
                    filetypes=[("CSV files", "*.csv")]
                )
                if not csv_file:
                    return False
            
            self.df = pd.read_csv(csv_file)
            self.df.columns = self.df.columns.str.strip()

            required_columns = self.feature_names + ['Salary']
            missing = [col for col in required_columns if col not in self.df.columns]
            if missing:
                messagebox.showerror("Missing Columns", f"Missing: {', '.join(missing)}")
                return False

            for col in ['Gender', 'Education Level', 'Location', 'Job Title', 'Skills']:
                self.encoders[col] = LabelEncoder()
                self.df[col] = self.encoders[col].fit_transform(self.df[col])

            messagebox.showinfo("Success", f"{len(self.df)} records loaded successfully.")
            return True
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return False
    
    def train_model(self):
        try:
            X = self.df[self.feature_names]
            y = self.df['Salary']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5,
                                               min_samples_leaf=2, random_state=42)
            self.model.fit(X_train_scaled, y_train)

            y_pred = self.model.predict(X_test_scaled)
            importance = list(zip(self.feature_names, self.model.feature_importances_))
            importance.sort(key=lambda x: x[1], reverse=True)

            self.model_performance = {
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'model_type': 'Random Forest',
                'feature_importance': importance
            }

            self.model_trained = True
            self.save_model()
            return True
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            return False

    def save_model(self):
        try:
            joblib.dump(self.model, 'salary_model.pkl')
            joblib.dump(self.scaler, 'salary_scaler.pkl')
            joblib.dump(self.encoders, 'salary_encoders.pkl')
        except Exception as e:
            print("Save failed:", e)

    def load_model(self):
        try:
            if os.path.exists('salary_model.pkl') and os.path.exists('salary_scaler.pkl') and os.path.exists('salary_encoders.pkl'):
                self.model = joblib.load('salary_model.pkl')
                self.scaler = joblib.load('salary_scaler.pkl')
                self.encoders = joblib.load('salary_encoders.pkl')
                self.model_trained = True
                return True
        except:
            return False
        return False

    def predict_salary(self, inputs):
        try:
            if not self.model_trained:
                messagebox.showwarning("Model Not Trained", "Please train or load a model first.")
                return None

            data = []
            for i, feature in enumerate(self.feature_names):
                if feature in ['Age', 'Years of Experience']:
                    data.append(float(inputs[i].get()))
                else:
                    val = inputs[i].get()
                    if val in self.encoders[feature].classes_:
                        encoded = self.encoders[feature].transform([val])[0]
                        data.append(encoded)
                    else:
                        raise ValueError(f"Invalid input for {feature}: {val}")

            scaled = self.scaler.transform([data])
            prediction = self.model.predict(scaled)[0]
            return round(prediction, 2)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return None

class SalaryPredictorGUI:
    def __init__(self):
        self.predictor = SalaryPredictor()
        self.setup_gui()
        if not self.predictor.load_model():
            self.train_new_model()

    def train_new_model(self):
        if self.predictor.load_and_prepare_data():
            if self.predictor.train_model():
                perf = self.predictor.model_performance
                msg = (
                    f"🌟 Model trained successfully!\n\n"
                    f"Train size: {perf['train_size']}\n"
                    f"Test size: {perf['test_size']}\n"
                    f"R² Score: {perf['r2']:.3f}\n"
                    f"MAE: ${perf['mae']:.2f}\n\n"
                    f"Top Features:\n"
                )
                for i, (feat, score) in enumerate(perf['feature_importance'][:5]):
                    msg += f"{i+1}. {feat} ({score:.2%})\n"
                messagebox.showinfo("Training Complete", msg)
                self.update_status()
                self.update_comboboxes()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Salary Predictor")
        self.root.attributes('-fullscreen', True)

        try:
            bg_img = Image.open('background_design.png')
            screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            bg_img = bg_img.resize((screen_w, screen_h))
            bg = ImageTk.PhotoImage(bg_img)
            bg_label = tk.Label(self.root, image=bg)
            bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)
            bg_label.image = bg
        except:
            self.root.configure(bg="#f0f0f0")

        self.form_frame = tk.Frame(self.root, bg="white", padx=40, pady=40)
        self.form_frame.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(self.form_frame, text="🌟 Random Forest Salary Predictor",
                 font=("Poppins", 24, "bold"), bg="white", fg="#4A00E0").grid(row=0, column=0, columnspan=3, pady=(0, 20))

        self.status_label = tk.Label(self.form_frame, text="Model Status: Loading...",
                                     font=("Poppins", 12), bg="white", fg="#666")
        self.status_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))

        self.create_input_fields()
        self.create_buttons()
        self.update_status()

    def create_input_fields(self):
        self.variables = []
        labels = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Location', 'Skills']

        for i, label in enumerate(labels):
            tk.Label(self.form_frame, text=label + ":", font=("Poppins", 13),
                     bg="white", fg="#333").grid(row=i+2, column=0, sticky="e", padx=10, pady=10)

            if label in ['Age', 'Years of Experience']:
                entry = tk.Entry(self.form_frame, font=("Poppins", 12), width=35)
            else:
                options = (sorted(self.predictor.encoders[label].classes_.tolist())
                           if self.predictor.model_trained and label in self.predictor.encoders
                           else ["Select..."])
                entry = ttk.Combobox(self.form_frame, values=options, font=("Poppins", 12), width=33, state="readonly")

            entry.grid(row=i+2, column=1, padx=10, pady=10)
            self.variables.append(entry)

    def create_buttons(self):
     # Load CSV Data button
     load_data_btn = tk.Button(self.form_frame, text="📁 Load CSV Data", command=self.train_new_model,
                              font=("Poppins", 14, "bold"), bg="#28a745", fg="white",
                              padx=20, pady=10, bd=0, activebackground="#218838", cursor="hand2")
     load_data_btn.grid(row=10, column=0, pady=(25, 10), padx=2, sticky="ew")

     # Predict Salary button
     predict_btn = tk.Button(self.form_frame, text="🔍 Predict Salary", command=self.predict_salary,
                            font=("Poppins", 14, "bold"), bg="#7f50ff", fg="white",
                            padx=20, pady=10, bd=0, activebackground="#5e36cc", cursor="hand2")
     predict_btn.grid(row=10, column=1, pady=(25, 10), padx=2, sticky="ew")

     # Exit button with red color
     exit_btn = tk.Button(self.form_frame, text="⏹ Exit", command=self.root.destroy,
                         font=("Poppins", 13), bg="#dc3545", fg="white",
                         padx=18, pady=8, bd=0, activebackground="#b02a37", cursor="hand2")
     exit_btn.grid(row=11, column=0, columnspan=2, pady=(10, 0), sticky="ew")



    def predict_salary(self):
        salary = self.predictor.predict_salary(self.variables)
        if salary is not None:
            messagebox.showinfo("Predicted Salary", f"💰 Estimated Salary: ${salary:,.2f}")

    def update_status(self):
        status = "Trained ✔️" if self.predictor.model_trained else "Not Trained ❌"
        self.status_label.config(text=f"Model Status: {status}")

    def update_comboboxes(self):
        for i, label in enumerate(self.predictor.feature_names):
            if label in ['Age', 'Years of Experience']:
                continue
            combo = self.variables[i]
            options = sorted(self.predictor.encoders[label].classes_.tolist())
            combo.config(values=options)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    SalaryPredictorGUI().run()