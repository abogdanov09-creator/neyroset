import tensorflow as tf
import numpy as np
import pandas as  pd


class Engineer:
    def __init__(self, alpha=0.001, data="file", excel_file="D:\Учебная Андрей/databuild.xlsx"):
        if data == "file":
            # Загрузка из Excel
            df = pd.read_excel(excel_file)

            # Смотрим что загрузили
            print("📊 Данные из Excel:")
            print(df.head())
            print(f"   Всего строк: {len(df)}")
            print(f"   Температура: от {df['temp'].min()} до {df['temp'].max()}")
            print(f"   Толщина: от {df['thickness'].min()} до {df['thickness'].max()}")

            # Превращаем в массивы для нейросети
            self.X_train = df[['temp', 'wall_type', 'class']].values
            self.y_train = df[['thickness']].values

        else:
            # Данные прямо в коде
            self.X_train = np.array([
                [-30, 0, 0], [-30, 0, 1], [-30, 0, 2],
                [-30, 1, 0], [-30, 1, 1], [-30, 1, 2],
                [-30, 2, 0], [-30, 2, 1], [-30, 2, 2],
                [-15, 0, 0], [-15, 0, 1], [-15, 0, 2],
                [-15, 1, 0], [-15, 1, 1], [-15, 1, 2],
                [-15, 2, 0], [-15, 2, 1], [-15, 2, 2],
                [0, 0, 0], [0, 0, 1], [0, 0, 2],
                [0, 1, 0], [0, 1, 1], [0, 1, 2],
                [0, 2, 0], [0, 2, 1], [0, 2, 2],
                [10, 0, 0], [10, 0, 1], [10, 0, 2],
                [10, 1, 0], [10, 1, 1], [10, 1, 2],
                [10, 2, 0], [10, 2, 1], [10, 2, 2],
            ], dtype=float)

            self.y_train = np.array([
                [200], [250], [300],
                [150], [200], [250],
                [100], [150], [200],
                [150], [200], [250],
                [100], [150], [200],
                [50], [100], [150],
                [100], [150], [200],
                [50], [100], [150],
                [0], [50], [100],
                [50], [100], [150],
                [0], [50], [100],
                [0], [0], [50],
            ], dtype=float)

            print(f"✅ Загружено {len(self.X_train)} строк из кода")

        self.alpha = alpha
        self.model = None

    def build(self, input_dmn=3, hidden_layers=[32, 64, 128]):
        self.model = tf.keras.Sequential()
    # 1 слой
        self.model.add(tf.keras.layers.Dense(hidden_layers[0], input_shape=[input_dmn]))
    # последующие слои
        for neurons in hidden_layers[1:]:
            self.model.add(tf.keras.layers.Dense(neurons, activation="relu"))
    # Последний слой
        self.model.add(tf.keras.layers.Dense(1))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
            loss='mse'
        )
    def train(self, epochs = 500):
        if self.model is None:
            self.build()

        self.model.fit(self.X_train, self.y_train, epochs=epochs, verbose=0)


    def predict(self, temp, wall_type, energy_class):
        if self.model is None:
            self.train()
        x = np.array([[temp, wall_type, energy_class]], dtype=float)
        result = self.model.predict(x, verbose=0)
        return result[0][0]
eng = Engineer()
print(eng.predict(-12, 1, 0))