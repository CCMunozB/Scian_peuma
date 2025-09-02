import os
import re
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import get_window
from antropy import perm_entropy
from scipy.stats import entropy
import time

from mne.io import read_raw_edf
from mne.time_frequency import psd_array_multitaper

import zipfile

def extraer_id_peuma1(ruta):
    # Utiliza una expresión regular para encontrar el número entre 'SF' y '_all'
    coincidencia = re.search(r'SF(\d+)_all', ruta)
    if coincidencia:
        return coincidencia.group(1)  # Retorna el número encontrado
    else:
        return None  # Retorna None si no se encuentra el ID
    
def extraer_id_peuma2(ruta):
    # Utiliza una expresión regular para encontrar el número entre 'MAT2/' y '_all'
    coincidencia = re.search(r'MAT2/(\d+-\d+)_all', ruta)
    if coincidencia:
        return coincidencia.group(1)  # Retorna el número encontrado
    else:
        return None  # Retorna None si no se encuentra el ID

def divide_in_windows(eeg_signal, fs, window_size, overlap):
        """
        Divide una señal EEG en ventanas solapadas de un tamaño especificado.

        Args:
            eeg_signal (np.ndarray o list): Señal EEG original (1D) que se va a dividir en ventanas.
            fs (int): Frecuencia de muestreo de la señal EEG en Hz (muestras por segundo).
            window_size (int): Tamaño de cada ventana en número de muestras.
            overlap (float): Proporción de solapamiento entre ventanas consecutivas (valor entre 0 y 1).

        Returns:
            windows (list of np.ndarray): Lista de ventanas solapadas, cada una de tamaño `window_size`.
        """
        step_size = int(window_size * (1 - overlap))  # tamaño del paso para aplicar el solapamiento
        windows = []
        for start in range(0, len(eeg_signal) - window_size + 1, step_size):
            windows.append(eeg_signal[start:start + window_size])
        return windows
    
def extraer_id_zip(path, peuma1=False):
    """
    Extrae el ID del paciente de las rutas para los archivos .zip (edf).
    - Si peuma1=False: extrae la secuencia con el patrón '####-####'.
    - Si peuma1=True: extrae el número después de 'SF' y antes del primer '-'.
    """
    if not peuma1:
        match = re.search(r'\d+-\d+', path)
        if match:
            return match.group()
        else:
            return None
    else:
        match = re.search(r'SF(\d+)-', path)
        if match:
            return match.group(1)
        else:
            return None
    
    
def unir_edf(zip_path, peuma1=False):
    """
    Descomprime un archivo ZIP que contiene archivos EDF, ordena los archivos por su hora de medición,
    concatena la señal del tercer canal EEG de todos los archivos y retorna el ID del paciente, 
    la frecuencia de muestreo y la señal concatenada.

    Parameters:
    - zip_path (str): Ruta al archivo ZIP que contiene los archivos .edf.

    Returns:
    - id_zip (str): Identificador del paciente extraído del nombre del archivo ZIP.
    - fs (float): Frecuencia de muestreo (Hz) de los archivos EDF, se asume constante para todos los archivos.
    - total_eeg (np.ndarray): Array unidimensional que contiene la concatenación de las señales del tercer canal de todos los archivos EDF.
    """
    # Extraer el ID del paciente y crear directorio de destino
    id_zip = extraer_id_zip(zip_path, peuma1)
    output_dir = f'edf_files/-{id_zip}/'

    print(extraer_id_zip(zip_path))

    # Descomprimir el ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Obtener la lista de rutas de todos los archivos descomprimidos (.edf)
    file_paths = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    file_paths = [path for path in file_paths if path.lower().endswith('.edf')]

    # Mostrar las rutas encontradas
    for path in file_paths:
        print(path)

    # Inicializar diccionario con ruta: hice de inicio del edf
    dict_edf = {}

    # Iterar sobre los archivos .edf de la carpeta
    for ruta in file_paths:
        try:
            # Leer archivo edf
            raw = read_raw_edf(ruta, eog=None, misc=None, stim_channel='auto', exclude=(), preload=True, verbose=False)

            # Extraer fecha y hora
            meas_date = raw.info['meas_date'] # datetime.datetime(2024, 9, 10, 7, 42, 6, tzinfo=datetime.timezone.utc)

            # Agregar la ruta como llave al diccionario, su valor es la hora
            dict_edf[ruta] = meas_date.time()
        except Exception as e:
            print(f'No fue posible leer el archivo. Error {e}')

    # Ordenar el diccionario por hora
    dict_edf = dict(sorted(dict_edf.items(), key=lambda x: x[1]))

    # Leer los archivos en orden e ir concatenando la señal del tercer canal
    fs_list = []
    total_eeg = np.array([])

    for ruta, hora in dict_edf.items():
        # Leer archivo
        raw = read_raw_edf(ruta, eog=None, misc=None, stim_channel='auto', exclude=(), preload=True, verbose=False)

        # Extraer frecuencia de muestreo
        fs = raw.info['sfreq']
        fs_list.append(fs)

        # Extraer señal del tercer canal
        #eeg_data = raw[2][0][0]  # Tercer canal de EEG

        # Extraer tercer canal con datos en microvoltios ESTE ES EL CAMBIO
        data_uV = raw.get_data(units='uV')  # ndarray shape (n_channels, n_times)
        if data_uV.shape[0] < 3:
            raise ValueError(f"El archivo {ruta} tiene menos de 3 canales.")
        eeg_data = data_uV[2]  # tercer canal

        total_eeg = np.concatenate((total_eeg, eeg_data))

    assert all(x == fs_list[0] for x in fs_list), "No todas las Fs son iguales"
    fs = fs_list[0]

    return id_zip, fs, total_eeg

def shannon_entropy(eeg_segment, num_bins=256):
    """
    Calcula la entropía de Shannon de un segmento de EEG utilizando scipy.stats.entropy.
    
    eeg_segment: segmento de la señal EEG (numpy array).
    num_bins: número de bins para la discretización.
    
    Returns:
    entropy_value: valor de la entropía de Shannon.
    """
    # Calcular el histograma de la señal EEG
    hist, _ = np.histogram(eeg_segment, bins=num_bins, density=True)
    
    # Eliminar ceros para evitar log(0)
    hist = hist[hist > 0]
    
    # Calcular la entropía de Shannon usando scipy.stats.entropy
    entropy_value = entropy(hist, base=2)
    
    return entropy_value


def calculate_permutation_entropy(eeg_segment, order=3, delay=1):
    """
    Calcula la entropía de permutación de un segmento de EEG.
    
    eeg_segment: segmento de la señal EEG (numpy array).
    order: tamaño de embedding (número de puntos a considerar en cada permutación).
    delay: retraso entre los puntos de la serie temporal.
    
    Returns:
    perm_entropy_value: valor de la entropía de permutación.
    """
    # Calcular la entropía de permutación
    perm_entropy_value = perm_entropy(eeg_segment, order=order, delay=delay)
    
    return perm_entropy_value


def calcular_synch_fast_slow(eeg_signal, window_type='hann', nfft=128, overlap=0.5):
    """
    Calcula el SynchFastSlow a partir del biespectro.
    
    eeg_signal (np.array): La señal EEG del tercer canal.
    window_type: tipo de ventana a aplicar
    nfft: número de puntos para la FFT
    overlap: fracción de superposición entre segmentos

    Returns:
    - synch_fast_slow (float): Valor del SynchFastSlow.
    """
    # Calcular el tamaño de la ventana y el paso
    window_size = nfft*5 # No necesitamos tantas ventanas, pues se calcula sobre una ventana
    step_size = int(window_size * (1 - overlap))
    
    # Obtener la ventana
    window = get_window(window_type, window_size)
    
    # Número de segmentos
    M = (len(eeg_signal) - window_size) // step_size + 1
    
    # Inicializar el biespectro
    biespec = np.zeros((nfft, nfft), dtype=complex)
    
    # Calcular el biespectro
    for k in range(M):
        start = k * step_size
        segment = eeg_signal[start:start + window_size] * window
        
        # Transformada de Fourier del segmento
        X_k = np.fft.fft(segment, n=nfft)
        
        for f1 in range(nfft):
            for f2 in range(nfft):
                f3 = f1 + f2
                if f3 < nfft:  # Asegurarse de que f3 esté dentro del rango
                    biespec[f1, f2] += X_k[f1] * X_k[f2] * np.conj(X_k[f3])
    
    # Promediar el biespectro
    biespec /= M

    # Calcular el parámetro SynchFastSlow a partir del biespectro

    # Sumar los picos del biespectro en el rango de 0.5 a 47 Hz
    sum_wide = np.sum(np.abs(biespec[1:48, 1:48]))  # Rango de 0.5 a 47 Hz
    # Sumar los picos del biespectro en el rango de 40 a 47 Hz
    sum_narrow = np.sum(np.abs(biespec[40:48, 40:48]))  # Rango de 40 a 47 Hz

    if sum_narrow > 0:
        synch_fast_slow = np.log(sum_wide / sum_narrow)
    else:
        synch_fast_slow = 0  # Manejo de caso donde la suma en el rango estrecho es cero

    return synch_fast_slow


def calcular_betaratio(eeg_signal, fs, nfft=128, window_type='hann'):
    """
    Calcula el BetaRatio de una señal EEG en base a la FFT.
    
    Parameters:
    - eeg_signal: la señal EEG (numpy array)
    - fs (float): frecuencia de muestreo
    - nfft (int): número de puntos para la FFT
    - window_type (str): tipo de ventana a aplicar
    
    Returns:
    - beta_ratio (float): Valor del BetaRatio.
    """
    # Obtener la ventana
    window = get_window(window_type, len(eeg_signal))
    
    # Calcular la FFT de la señal completa
    segment = eeg_signal * window  # Aplicar la ventana a la señal
    fft_values = np.fft.fft(segment, n=nfft)
    
    # Obtener las frecuencias correspondientes
    freqs = np.fft.fftfreq(nfft, d=1/fs)
    
    # Calcular la potencia en las bandas de frecuencia
    P30_47 = np.sum(np.abs(fft_values[(freqs >= 30) & (freqs <= 47)])**2)  # Potencia en la banda 30-47 Hz
    P11_20 = np.sum(np.abs(fft_values[(freqs >= 11) & (freqs <= 20)])**2)  # Potencia en la banda 11-20 Hz
    
    # Calcular el BetaRatio
    if P11_20 > 0:
        beta_ratio = np.log(P30_47 / P11_20)
    else:
        beta_ratio = 0  # Manejo de caso donde la potencia en la banda 11-20 Hz es cero

    return beta_ratio


def generar_features_new(file_paths, window_size_seg=30, overlap=0.5, peuma1 = True, edf = False, aux_save_path=None): # <-- 1. PARÁMETRO NUEVO
    """
    Divide el tercer canal del EEG en ventanas con traslape y calcula una serie de features para cada ventana.
    Opcionalmente, guarda un archivo CSV por cada paciente procesado en `aux_save_path`.
    """
    # Títulos de las características a extraer
    feat_title = ['name_window', 'tmin', 'tmax', 'tmid', 'F1 Poder total (0.5 a 40 Hz)', 'F2 Alpha-power (9 a 12 Hz)', 'F3 Relative-alpha (9 a 12 Hz)','F4 Alpha peak',
                'F5 Alpha-power (8 a 13 Hz)', 'F6 Relative-alpha (8 a 13 Hz)', 'F7 Poder relativo delta (0.5 a 4 Hz)', 'F8 Poder relativo theta (4 a 8 Hz)',
                'F9 Poder relativo beta (15 a 30 Hz)', 'F10 Intensidad promedio abs en cada ventana ', 'F11 Shannon entropy', 'F12 Permutation entropy', 'F13 SynchFastSlow', 'F14 BetaRatio']
    
    # Lista para guardar los datos de TODOS los pacientes
    datos_consolidados = [feat_title]

    # Iterar sobre las rutas de los archivos
    start = time.time()
    for file_path in file_paths:
        id_paciente = None # Inicializar para un mejor manejo de errores
        try:
            if not edf:
                # Lectura de archivo .mat
                data = scipy.io.loadmat(file_path)
                eeg = data['catEEG']

                # Extraer frecuencia de muestreo
                fs_array = data['Firsthdr'][0][0][-1].flatten()
                fs = fs_array[2]

                # Filtramos la señal (entre 0.5 y 40 Hz)
                fmin, fmax = 0.5, 40
                b, a = scipy.signal.butter(N = 4, Wn = [fmin / (fs/2), fmax / (fs/2)], btype = 'band')
                eeg_filtered = scipy.signal.filtfilt(b, a, eeg[2, :]) # Se usa el tercer canal
    
                # Guardamos la señal entre 0.5 y 47 Hz, pues BetaRatio y SynchFastSlow consideran frecuencias en ese rango
                if fs/2 < 47:
                    b, a = scipy.signal.butter(N = 4, Wn = [0.5 / (fs/2), 0.99], btype = 'band')
                else:
                    b, a = scipy.signal.butter(N = 4, Wn = [0.5 / (fs/2), 47 / (fs/2)], btype = 'band')
                eeg_filtered_for_betaratio = scipy.signal.filtfilt(b, a, eeg[2, :])

                # ID del paciente
                if peuma1:
                    id_paciente = extraer_id_peuma1(file_path)
                else:
                    id_paciente = extraer_id_peuma2(file_path)
            
            else:
                # Lectura de archivos .edf en el zip y unión de los segmentos del EEG
                id_paciente, fs, eeg_data = unir_edf(file_path, peuma1)

                # Filtramos la señal (entre 0.5 y 40 Hz)
                fmin, fmax = 0.5, 40
                b, a = scipy.signal.butter(N = 4, Wn = [fmin / (fs/2), fmax / (fs/2)], btype = 'band')
                eeg_filtered = scipy.signal.filtfilt(b, a, eeg_data) 

                # Guardamos la señal entre 0.5 y 47 Hz
                if fs/2 < 47:
                    b, a = scipy.signal.butter(N = 4, Wn = [0.5 / (fs/2), 0.99], btype = 'band')
                else:
                    b, a = scipy.signal.butter(N = 4, Wn = [0.5 / (fs/2), 47 / (fs/2)], btype = 'band')
                eeg_filtered_for_betaratio = scipy.signal.filtfilt(b, a, eeg_data)

            # Dividimos la data en ventanas
            window_size = int(fs * window_size_seg)
            windows = divide_in_windows(eeg_filtered, fs, window_size, overlap)
            windows_beta_ratio = divide_in_windows(eeg_filtered_for_betaratio, fs, window_size, overlap)

            # Lista temporal para guardar las filas SOLO del paciente actual
            datos_paciente_actual = []
            n = 0
            for i in range(len(windows)):
                window = windows[i]
                window_beta_ratio = windows_beta_ratio[i]

                # Guardamos tmin y tmax (tiempo de incio y fin) en segundos, junto al tiempo medio de la ventana
                tmin = window_size_seg * n * (1 - overlap)
                tmax = window_size_seg * (1 + n * (1 - overlap))
                tmid = tmin + (tmax - tmin) / 2

                # Cálculo de la densidad espectral de potencia (PSD) con el método MultiTaper
                psds, freqs = psd_array_multitaper(window[np.newaxis, :], sfreq = fs, fmin = fmin, fmax = fmax,
                                                bandwidth = 2, low_bias = True, n_jobs = 3, verbose=False)
                psds_all_freq = psds.flatten()

                # F1 potencia promedio
                F1 = np.mean(psds_all_freq)
                # F2 potencia media en la banda alpha (9-12 Hz)
                potencias_9_12 = psds_all_freq[(freqs >= 9) & (freqs <= 12)]
                F2 = np.mean(potencias_9_12)
                # F3 potencia relativa en la banda alpha (9-12 Hz)
                poder_total = np.trapz(psds_all_freq, freqs)
                F3 = np.trapz(potencias_9_12, freqs[np.where((freqs >= 9) & (freqs <= 12))]) / poder_total
                # F4 frecuencia con máxima potencia en la banda alpha (alpha peak)
                F4 = freqs[np.where((freqs >= 9) & (freqs <= 12))][np.argmax(potencias_9_12)]
                # F5 y F6 potencia banda alpha (8-13 Hz) (media y relativa)
                potencias_8_13 = psds_all_freq[(freqs >= 8) & (freqs <= 13)]
                F5 = np.mean(potencias_8_13)
                F6 = np.trapz(potencias_8_13, freqs[np.where((freqs >= 8) & (freqs <= 13))]) / poder_total
                # F7 potencia relativa en la banda delta (0.5-4 Hz)
                potencias_05_4 = psds_all_freq[(freqs >= 0.5) & (freqs <= 4)]
                F7 = np.trapz(potencias_05_4, freqs[np.where((freqs >= 0.5) & (freqs <= 4))]) / poder_total
                # F8 potencia relativa en la banda theta (4-8 Hz)
                potencias_4_8 = psds_all_freq[(freqs >= 4) & (freqs <= 8)]
                F8 = np.trapz(potencias_4_8, freqs[np.where((freqs >= 4) & (freqs <= 8))]) / poder_total
                # F9 potencia relativa en la banda beta (15-30 Hz)
                potencias_15_30 = psds_all_freq[(freqs >= 15) & (freqs <= 30)]
                F9 = np.trapz(potencias_15_30, freqs[np.where((freqs >= 15) & (freqs <= 30))]) / poder_total
                # F10 promedio absoluto de la señal
                F10 = np.mean(abs(window))
                # F11 entropía de Shannon
                F11 = shannon_entropy(window)
                # F12 entropía de permutación
                F12 = calculate_permutation_entropy(window)
                # F13 SynchFastSlow
                F13 = calcular_synch_fast_slow(window_beta_ratio)
                # F14 BetaRatio
                F14 = calcular_betaratio(window_beta_ratio, fs)

                # Se agregan los datos a la lista temporal del paciente
                datos_paciente_actual.append([id_paciente + '_' + str(n), tmin, tmax, tmid, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14])
                n += 1

            # =================================================================
            # === INICIO DEL BLOQUE DE CÓDIGO AÑADIDO PARA GUARDADO AUXILIAR ===
            # =================================================================
            if aux_save_path and datos_paciente_actual:
                os.makedirs(aux_save_path, exist_ok=True)
                df_paciente = pd.DataFrame(datos_paciente_actual, columns=feat_title)
                ruta_archivo_aux = os.path.join(aux_save_path, f"{id_paciente}_features.csv")
                df_paciente.to_csv(ruta_archivo_aux, index=False)
            
            # Se añaden los datos del paciente procesado a la lista consolidada final
            datos_consolidados.extend(datos_paciente_actual)
            # ===============================================================
            # === FIN DEL BLOQUE DE CÓDIGO AÑADIDO ===
            # ===============================================================
            print(f"Tiempo de procesamiento del archivo {file_path}: {time.time() - start} segundos")
            start = time.time()  # Reiniciar el tiempo para el siguiente archivo
        except Exception as e:
            if id_paciente:
                print(f"No fue posible generar features para el ID del paciente: {id_paciente}. Error: {e}")
            else:
                if peuma1:
                    id_paciente = extraer_id_peuma1(file_path) if 'extraer_id_peuma1' in locals() else 'desconocido'
                else:
                    id_paciente = extraer_id_peuma2(file_path) if 'extraer_id_peuma2' in locals() else 'desconocido'
                print(f"No fue posible procesar el archivo para el paciente {id_paciente}. Error: {e}")

    features_df = pd.DataFrame(datos_consolidados[1:], columns=datos_consolidados[0])
    return features_df

def calcular_sef(psds, freqs, porcentaje):
    """
    Calcula la frecuencia de borde espectral (SEF) para un porcentaje dado
    
    Args:
        psds: densidad espectral de potencia (array)
        freqs: frecuencias correspondientes (array)
        porcentaje: porcentaje de potencia acumulada (ej. 90, 95, 50)
    
    Returns:
        SEF: frecuencia de borde espectral
    """
    # Calcula la potencia total
    potencia_total = np.sum(psds)
    
    # Calcula la potencia acumulada normalizada
    potencia_acumulada = np.cumsum(psds) / potencia_total
    
    # Encuentra el índice donde la potencia acumulada supera el porcentaje
    indice = np.argmax(potencia_acumulada >= porcentaje/100)
    
    # Si no se alcanza el porcentaje (poco probable), devolver la última frecuencia
    if indice == 0 and potencia_acumulada[0] >= porcentaje/100:
        return freqs[0]
    elif indice == 0:
        return freqs[-1]
    
    # Interpolación lineal para mayor precisión
    if indice < len(freqs):
        x = [potencia_acumulada[indice-1], potencia_acumulada[indice]]
        y = [freqs[indice-1], freqs[indice]]
        return np.interp(porcentaje/100, x, y)
    else:
        return freqs[-1]
    
def generar_sef_new(file_paths, window_size_seg=30, overlap=0.5, peuma1 = True, edf = False, aux_save_path=None): # <--- 1. PARÁMETRO NUEVO
    """
    Calcula el SEF 50, 90 y 95 y opcionalmente guarda un CSV por cada paciente procesado.
    """
    feat_title = ['name_window', 'tmin', 'tmax', 'tmid', 'SEF50', 'SEF90', 'SEF95']
    # Lista para guardar los datos de TODOS los pacientes
    datos_consolidados = [feat_title]
    
    start = time.time()
    for file_path in file_paths:
        id_paciente = None # Inicializar ID
        try:
            # Lógica para extraer el ID del paciente (sin cambios)
            if not edf:
                if peuma1:
                    id_paciente = extraer_id_peuma1(file_path)
                else:
                    id_paciente = extraer_id_peuma2(file_path)
                print(f'ID paciente: {id_paciente}')
            else:
                id_paciente, fs, eeg_data = unir_edf(file_path, peuma1)

            # Lógica para leer y filtrar la señal (sin cambios)
            if not edf:
                data = scipy.io.loadmat(file_path)
                eeg = data['catEEG']
                fs_array = data['Firsthdr'][0][0][-1].flatten()
                fs = fs_array[2]
                fmin, fmax = 0.5, 40
                b, a = scipy.signal.butter(N = 4, Wn = [fmin / (fs/2), fmax / (fs/2)], btype = 'band')
                eeg_filtered = scipy.signal.filtfilt(b, a, eeg[2, :])
            else:
                fmin, fmax = 0.5, 40
                b, a = scipy.signal.butter(N=4, Wn=[fmin / (fs/2), fmax / (fs/2)], btype='band')
                eeg_filtered = scipy.signal.filtfilt(b, a, eeg_data)

            # Lógica de división en ventanas (sin cambios)
            window_size = int(fs * window_size_seg)
            windows = divide_in_windows(eeg_filtered, fs, window_size, overlap)

            # Lista temporal para guardar las filas SOLO del paciente actual
            datos_paciente_actual = []
            n = 0
            for i in range(len(windows)):
                window = windows[i]

                # Guardamos tmin y tmax (tiempo de incio y fin) en segundos, junto al tiempo medio
                tmin = 30 * n * (1 - overlap)
                tmax = 30 * (1 + n * (1 - overlap))
                tmid = tmin + (tmax - tmin) / 2

                # Cálculo de la densidad espectral de potencia (PSD) con el método MultiTaper
                psds, freqs = psd_array_multitaper(window[np.newaxis, :], sfreq=fs, fmin=fmin, fmax=fmax,
                                                bandwidth=2, low_bias=True, n_jobs=3, verbose=False)
                psds_all_freq = psds.flatten()

                # Cálculo de SEFs
                sef50 = calcular_sef(psds_all_freq, freqs, 50)
                sef90 = calcular_sef(psds_all_freq, freqs, 90)
                sef95 = calcular_sef(psds_all_freq, freqs, 95)     

                # Se agregan los datos a la lista temporal del paciente
                datos_paciente_actual.append([id_paciente + '_' + str(n), tmin, tmax, tmid, sef50, sef90, sef95])
                n += 1

            # =================================================================
            # === INICIO DEL BLOQUE DE CÓDIGO AÑADIDO PARA GUARDADO AUXILIAR ===
            # =================================================================
            if aux_save_path and datos_paciente_actual:
                os.makedirs(aux_save_path, exist_ok=True)
                df_paciente = pd.DataFrame(datos_paciente_actual, columns=feat_title)
                ruta_archivo_aux = os.path.join(aux_save_path, f"{id_paciente}_sef.csv")
                df_paciente.to_csv(ruta_archivo_aux, index=False)
            
            # Se añaden los datos del paciente procesado a la lista consolidada final
            datos_consolidados.extend(datos_paciente_actual)
            # ===============================================================
            # === FIN DEL BLOQUE DE CÓDIGO AÑADIDO ===
            # ===============================================================
            print(f"Tiempo de procesamiento del archivo {file_path}: {time.time() - start} segundos")
            start = time.time()  # Reiniciar el tiempo para el siguiente archivo

        except Exception as e:
            # Ahora 'id_paciente' tendrá un valor aquí (si la extracción fue exitosa)
            if id_paciente:
                print(f"No fue posible generar SEF para el ID del paciente: {id_paciente}. Error: {e}")
            else:
                print(f"No fue posible procesar el archivo: {file_path}. Error: {e}")


    features_df = pd.DataFrame(datos_consolidados[1:], columns=datos_consolidados[0])
    return features_df
