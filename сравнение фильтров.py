import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

np.random.seed(26)

# Генерация синтетического сигнала
def generate_signal(seq_length=1000):
    time = np.linspace(0, seq_length, seq_length)

    # Основной синусоидальный сигнал
    base_frequency = 1
    signal = np.sin(2 * np.pi * base_frequency * time / (5 * seq_length))

    # Добавление гармоник
    for A, f_mult in ((0.5, 3), (0.3, 5)):
        harmonics = A * np.sin(2 * np.pi * (base_frequency * f_mult) * time / (6 * seq_length))
        signal += harmonics

    # Добавление случайных импульсов
    for _ in range(40): 
        impulse_position = np.random.randint(0, seq_length)
        signal[impulse_position] += np.random.uniform(7, 10)

    # Добавление шума
    noise = np.random.normal(0, 0.5, seq_length)
    signal += noise

    return signal

# Функция для применения фильтра Баттерворта
def butter_filter(data, cutoff, fs, btype='low', order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = lfilter(b, a, data)
    return y

# Нормализация сигнала
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Параметры сигнала и фильтров
fs = 100  # частота дискретизации
signal_length = 1000
signal = generate_signal(signal_length)

# Фильтры
lowcut = 5.0  # Низкочастотный фильтр
highcut = 10.0 # Высокочастотный фильтр
bandpasscut = [5.0, 10.0]  # Полосовой фильтр
gaussian_filter = np.exp(-np.linspace(-3, 3, 21)**2)  # Гауссовский фильтр

# Применяем фильтры Баттерворта
low_filtered = butter_filter(signal, lowcut, fs, btype='low')
high_filtered = butter_filter(signal, highcut, fs, btype='high')
bandpass_filtered = butter_filter(low_filtered - high_filtered, bandpasscut[0], fs, btype='low')
gaussian_filtered = np.convolve(signal, gaussian_filter/gaussian_filter.sum(), mode='same')

# Нормируем сигналы
signal_normalized = normalize(signal)
low_filtered_normalized = normalize(low_filtered)
high_filtered_normalized = normalize(high_filtered)
bandpass_filtered_normalized = normalize(bandpass_filtered)
gaussian_filtered_normalized = normalize(gaussian_filtered)

# Визуализация результатов
plt.figure(figsize=(12, 8))

plt.plot(signal_normalized, label='Сигнал + Шум', color='red', alpha=0.5, linestyle='-')
plt.plot(low_filtered_normalized, label='ФНЧ', color='yellow', linestyle='--')
plt.plot(high_filtered_normalized, label='ФВЧ', color='blue', linestyle='-.')
plt.plot(bandpass_filtered_normalized, label='Полосовой', color='orange', linestyle=':')
plt.plot(gaussian_filtered_normalized, label='Гаусовский', color='green', linestyle='-.')

plt.xlabel('Отсчеты')
plt.ylabel('А (нормированная)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
