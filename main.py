import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

class TimeCell:
    def __init__(self, date):
        self.date = pd.to_datetime(date)
        self.day = self.date.day
        self.month = self.date.month
        self.year = self.date.year

class WeatherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hava Durumu Tahmin Uygulaması")

        self.frame = tk.Frame(root, padx=10, pady=10)
        self.frame.pack(padx=10, pady=10)

        self.load_button = tk.Button(self.frame, text="Veri Seti Yükle", command=self.load_dataset, bg="blue", fg="white")
        self.load_button.grid(row=0, column=0, padx=10, pady=10)

        self.run_button = tk.Button(self.frame, text="Modeli Çalıştır", command=self.run_model, bg="green", fg="white")
        self.run_button.grid(row=0, column=1, padx=10, pady=10)

        self.result_label = tk.Label(self.frame, text="", wraplength=400, justify="left", font=("Arial", 12))
        self.result_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    def load_dataset(self):
        self.filepath = filedialog.askopenfilename()
        if self.filepath:
            self.data = pd.read_csv(self.filepath)
            self.result_label.config(text="Veri seti yüklendi!")
        else:
            messagebox.showerror("Hata", "Dosya seçilmedi.")

    def run_model(self):
        try:
            self.data = self.data.ffill()

            self.data['timecell'] = self.data['date'].apply(TimeCell)

            y_temp_max = self.data['temp_max']
            y_temp_min = self.data['temp_min']
            y_weather = self.data['weather']

            label_encoder = LabelEncoder()
            y_weather_encoded = label_encoder.fit_transform(y_weather)

            X = self.data[['precipitation', 'wind']].copy()
            X['day'] = self.data['timecell'].apply(lambda x: x.day)
            X['month'] = self.data['timecell'].apply(lambda x: x.month)
            X['year'] = self.data['timecell'].apply(lambda x: x.year)

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train_max, y_test_max, y_train_min, y_test_min, y_train_weather, y_test_weather = train_test_split(
                X_scaled, y_temp_max, y_temp_min, y_weather_encoded, test_size=0.2, random_state=42)

            model_max = Sequential()
            model_max.add(Input(shape=(X_train.shape[1],)))
            model_max.add(Dense(64, activation='relu'))
            model_max.add(Dense(32, activation='relu'))
            model_max.add(Dense(1))
            model_max.compile(loss='mean_squared_error', optimizer='adam')

            model_min = Sequential()
            model_min.add(Input(shape=(X_train.shape[1],)))
            model_min.add(Dense(64, activation='relu'))
            model_min.add(Dense(32, activation='relu'))
            model_min.add(Dense(1))
            model_min.compile(loss='mean_squared_error', optimizer='adam')

            model_weather = Sequential()
            model_weather.add(Input(shape=(X_train.shape[1],)))
            model_weather.add(Dense(64, activation='relu'))
            model_weather.add(Dense(32, activation='relu'))
            model_weather.add(Dense(len(label_encoder.classes_), activation='softmax'))
            model_weather.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

            model_max.fit(X_train, y_train_max, epochs=50, batch_size=10)
            model_min.fit(X_train, y_train_min, epochs=50, batch_size=10)
            model_weather.fit(X_train, y_train_weather, epochs=50, batch_size=10)

            predictions_max = model_max.predict(X_test)
            predictions_min = model_min.predict(X_test)
            predictions_weather = model_weather.predict(X_test)
            predictions_weather_labels = label_encoder.inverse_transform(np.argmax(predictions_weather, axis=1))

            weather_translation = {
                "rain": "Yağmurlu",
                "sun": "Güneşli",
                "snow": "Karlı",
            }

            future_dates = pd.date_range(start="2024-07-01", periods=len(predictions_max), freq='D')

            result_str = ""
            for i in range(len(predictions_max)):
                date = future_dates[i].strftime("%d.%m.%Y")
                max_temp = f"{predictions_max[i][0]:.3f}"
                min_temp = f"{predictions_min[i][0]:.3f}"
                weather = weather_translation.get(predictions_weather_labels[i], predictions_weather_labels[i])
                result_str += f"Tarih: {date} \nMaksimum Tahmini Hava Sıcaklığı: {max_temp}°C \nMinimum Tahmini Hava Sıcaklığı: {min_temp}°C \nTahmini Hava Durumu: {weather}\n\n"

            self.result_label.config(text=result_str)
        except Exception as e:
            messagebox.showerror("Hata", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()
