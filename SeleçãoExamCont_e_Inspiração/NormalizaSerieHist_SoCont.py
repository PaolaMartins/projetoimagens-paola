import os
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd


# ------------------------- CONFIGURAÇÕES ----------------------------
root_folder = r"C:\Users\Home\Documents\USP2024\IC\Pacientes"
series_descriptions_insp = [
    'CONTRASTE', 'CONTRASTE VENOSO'
]
scores = {desc: len(series_descriptions_insp) - i for i, desc in enumerate(series_descriptions_insp)}
window_center = 40
window_width = 400

# ---------------------- FUNÇÕES AUXILIARES --------------------------
def extract_studies_metadata(root_folder):
    all_data = {}
    for paciente in os.listdir(root_folder):
        paciente_path = os.path.join(root_folder, paciente)
        if not os.path.isdir(paciente_path):
            continue
        studies = []
        for estudo_uid in os.listdir(paciente_path):
            estudo_path = os.path.join(paciente_path, estudo_uid)
            if not os.path.isdir(estudo_path):
                continue
            series_list = []
            for serie_uid in os.listdir(estudo_path):
                serie_path = os.path.join(estudo_path, serie_uid)
                if not os.path.isdir(serie_path):
                    continue
                dicoms = [f for f in os.listdir(serie_path) if f.endswith('.dcm')]
                if not dicoms:
                    continue
                try:
                    ds = pydicom.dcmread(os.path.join(serie_path, dicoms[0]), stop_before_pixels=True)
                    series_list.append({
                        "Series Description": ds.SeriesDescription if 'SeriesDescription' in ds else 'ND',
                        "Series Instance UID": ds.SeriesInstanceUID,
                        "Number of Slices": len(dicoms)
                    })
                except Exception as e:
                    print(f"Erro ao ler {serie_path}: {e}")
            if series_list:
                study_date = ds.StudyDate if 'StudyDate' in ds else '00000000'
                studies.append({
                    "Study Date": study_date,
                    "Study Instance UID": estudo_uid,
                    "Series": series_list
                })
        if studies:
            all_data[paciente] = studies
    return all_data

def selectInspSeries(series):
    for s in series:
        s['score'] = scores.get(s.get('Series Description', ''), -1)
    score_candidatas = [s for s in series if s['score'] >= 0]
    if not score_candidatas:
        return None
    max_score = max(s['score'] for s in score_candidatas)
    top_scored = [s for s in score_candidatas if s['score'] == max_score]
    return max(top_scored, key=lambda s: s.get('Number of Slices', 0))

def applyWindowSeries(array):
    window_min = window_center - (window_width / 2)
    window_max = window_center + (window_width / 2)
    array = np.clip(array, window_min, window_max)
    return (array - window_min) / (window_max - window_min) * 255

def loadSeries(folder_path):
    slices = [pydicom.dcmread(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    ds = slices[0]
    volume = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    volume = volume * ds.RescaleSlope + ds.RescaleIntercept
    volume = applyWindowSeries(volume)
    return sitk.GetImageFromArray(volume)

def selectLoadSeries(study, study_uid, root_folder, patient_id):
    study_insp_serie = selectInspSeries(study['Series'])
    if study_insp_serie is None:
        print(f"Nenhuma série de inspiração encontrada para {patient_id}, estudo {study_uid}")
        return None
    series_uid = study_insp_serie['Series Instance UID']
    series_path = os.path.join(root_folder, patient_id, study_uid, series_uid)
    print(f"-Série selecionada para {patient_id} - {study['Study Date']} | {study_insp_serie['Series Description']}")
    return loadSeries(series_path), series_path, study_insp_serie

def loadPatientSeries(patient_id, studies, root_folder, prog_log):
    series = []
    for study in studies:
        study_uid = study['Study Instance UID']
        result = selectLoadSeries(study, study_uid, root_folder, patient_id)
        if result is not None:
            serie_img, serie_path, metadata = result
            series.append((serie_img, metadata))
    return series

def extract_histogram_features(volume_array, patient_id, series_id, series_description, has_contrast):
    array = volume_array.flatten()

    array = array[array !=0] 

    features = {
        "PatientID": patient_id,
        "SeriesID": series_id,
        "SeriesDescription": series_description,
        "Mean": np.mean(array),
        "Median": np.median(array),
        "StdDev": np.std(array),
        "Min": np.min(array),
        "Max": np.max(array),
        "Skewness": stats.skew(array),
        "Kurtosis": stats.kurtosis(array),
        "Percentile_25": np.percentile(array, 25),
        "Percentile_75": np.percentile(array, 75),
        "Entropy": stats.entropy(np.histogram(array, bins=100)[0] + 1)  
    }
    return features

# -------------------- EXECUÇÃO PRINCIPAL ----------------------------
pacientes_data = extract_studies_metadata(root_folder)
hist_folder = os.path.join(root_folder, "Histogramas")
os.makedirs(hist_folder, exist_ok=True)

features_list = []

for patient_id, studies in pacientes_data.items():
    prog_log = []
    series = loadPatientSeries(patient_id, studies, root_folder, prog_log)
    print(f"Paciente: {patient_id}, {len(series)} séries válidas carregadas.")

    for i, (volume, meta) in enumerate(series):
        array = sitk.GetArrayFromImage(volume).flatten()

        array = array[array != 0]
        features = extract_histogram_features(array, patient_id, f"Serie{i + 1}", meta["Series Description"],
                                              has_contrast=True or False)
        features_list.append(features)
        plt.figure(figsize=(8, 4))
        plt.hist(array, bins=100, color='steelblue', edgecolor='black', log=True)
        plt.title(f"Histograma - Paciente {patient_id} - Série {i+1}\n{meta['Series Description']}")
        plt.xlabel("Intensidade (1–255)")
        plt.ylabel("Frequência")
        plt.tight_layout()
        filename = f"{patient_id}_serie_{i+1}_histograma.png"
        plt.savefig(os.path.join(hist_folder, filename))
        plt.close()
        print(f" Histograma salvo: {filename}")

df_features = pd.DataFrame(features_list)
df_features.to_csv("features_histogramasSoCont.csv", index=False)
