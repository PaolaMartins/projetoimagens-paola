import os
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import SimpleITK as sitk

#36136 nro processo
# ===================================================================
# CONFIGURAÇOES
root_folder = r"C:\Users\Home\Documents\USP2024\IC\Pacientes"

planilha_contraste = r"C:\Users\Home\Documents\USP2024\IC\Codigos\SeleçãoExamCont_e_Inspiração\Teste_UIDs_Series\ct_torax_contraste_NumeroExames.csv"
planilha_sem_contraste = r"C:\Users\Home\Documents\USP2024\IC\Codigos\SeleçãoExamCont_e_Inspiração\Teste_UIDs_Series\ct_torax_SC_NumeroExames.csv"

features_path =r"C:\Users\Home\Documents\USP2024\IC\Codigos\SeleçãoExamCont_e_Inspiração\features_teste.csv"

window_center = 40   # ajuste conforme necessário
window_width = 400   # ajuste conforme necessário
# ===================================================================

# Carrega UIDs das planilhas
df1 = pd.read_csv(planilha_contraste, sep=";", dtype=str)
uids_contraste = df1["_id"].tolist()

df2 = pd.read_csv(planilha_sem_contraste, sep=";", dtype=str)
uids_sem_contraste = df2["_id"].tolist()

# ===================================================================
# FUNÇÕES AUXILIARES

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
                        "Series Description": getattr(ds, "SeriesDescription", "ND"),
                        "Series Instance UID": ds.SeriesInstanceUID,
                        "Number of Slices": len(dicoms),
                        "Rows": getattr(ds, "Rows", None),
                        "Columns": getattr(ds, "Columns", None),
                        "Folder Path": serie_path
                    })
                except Exception as e:
                    print(f"Erro ao ler {serie_path}: {e}", flush=True)
            if series_list:
                study_date = getattr(ds, "StudyDate", "00000000")
                studies.append({
                    "Study Date": study_date,
                    "Study Instance UID": estudo_uid,
                    "Series": series_list
                })
        if studies:
            all_data[paciente] = studies
    return all_data

def applyWindowSeries(array):
    window_min = window_center - (window_width / 2)
    window_max = window_center + (window_width / 2)
    array = np.clip(array, window_min, window_max)
    return (array - window_min) / (window_max - window_min) * 255

def loadSeries(folder_path):
    print(f"      > Carregando série em: {folder_path}", flush=True)
    slices = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(".dcm"):
            path = os.path.join(folder_path, f)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=False)
                slices.append(ds)
            except Exception as e:
                print(f"  !-Erro ao ler {f}: {e}", flush=True)
                continue

    if len(slices) == 0:
        print(f" Nenhum DICOM válido encontrado em {folder_path}", flush=True)
        return None

    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        print(f"   !-Série em {folder_path} sem ImagePositionPatient — ignorada.", flush=True)
        return None
    except Exception as e:
        print(f"   !-Erro inesperado ao ordenar {folder_path}: {e}", flush=True)
        return None

    ds = slices[0]
    try:
        volume = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    except Exception as e:
        print(f"Erro ao empilhar pixel_array em {folder_path}: {e}", flush=True)
        return None

    slope = getattr(ds, "RescaleSlope", 1)
    intercept = getattr(ds, "RescaleIntercept", 0)
    volume = volume * slope + intercept

    try:
        volume = applyWindowSeries(volume)
    except Exception as e:
        print(f"Erro ao aplicar janela em {folder_path}: {e}", flush=True)
        return None

    print(f"      - Série carregada com sucesso ({len(slices)} cortes)", flush=True)
    return sitk.GetImageFromArray(volume)

def extract_histogram_features(volume_array, patient_id, series_id, series_description, has_contrast):
    array = sitk.GetArrayFromImage(volume_array).flatten()
    array = array[array != 0]

    features = {
        "PatientID": patient_id,
        "SeriesID": series_id,
        "SeriesDescription": series_description,
        "HasContrast": has_contrast,
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

def select_best_series(series_list):
    """Seleciona série com maior número de slices e resolução 512x512."""
    best_series = None
    max_slices = -1
    for s in series_list:
        if s["Rows"] == 512 and s["Columns"] == 512:
            if s["Number of Slices"] > max_slices:
                max_slices = s["Number of Slices"]
                best_series = s
    return best_series

all_data = extract_studies_metadata(root_folder)
# ===================================================================
#                             LOOP PRINCIPAL
resultados = []

total_exames = 0
total_contraste = 0
total_sem_contraste = 0
series_validas = 0
sem_serie_valida = 0

for patient_id, studies in tqdm(all_data.items(), desc="Pacientes"):
    print(f"\nPaciente: {patient_id}", flush=True)

    for study in studies:
        study_uid = study["Study Instance UID"]
        if study_uid in uids_contraste:
            has_contrast = 1
            total_contraste += 1
        elif study_uid in uids_sem_contraste:
            has_contrast = 0
            total_sem_contraste += 1
        else:
            print(f"  Estudo {study_uid} não está nas planilhas — ignorado", flush=True)
            continue  # ignora se UID do estudo não está na planilha

        total_exames += 1
        print(f"  Processando estudo UID: {study_uid}, Contraste: {has_contrast}", flush=True)

        # Seleciona a melhor série
        best_series = select_best_series(study["Series"])
        if not best_series:
            sem_serie_valida += 1
            print(f"   !-Nenhuma série válida para estudo {study_uid}", flush=True)
            continue

        series_validas += 1
        print(f"   Série selecionada: {best_series['Series Instance UID']} "
              f"({best_series['Number of Slices']} cortes, {best_series['Rows']}x{best_series['Columns']})",
              flush=True)

        # Carrega volume
        volume = loadSeries(best_series["Folder Path"])
        if volume is None:
            print(f"   !-Falha ao carregar série {best_series['Series Instance UID']}", flush=True)
            continue

        # Calcula features
        feats = extract_histogram_features(
            volume,
            patient_id,
            best_series["Series Instance UID"],
            best_series["Series Description"],
            has_contrast
        )
        resultados.append(feats)

# SALVA RESULTADOS
df_out = pd.DataFrame(resultados)
df_out = df_out.round(3)
df_out.to_csv(features_path, index=False, sep=";", decimal=",")


print(f"\n- Finalizado! {len(resultados)} exames processados.", flush=True)
print(f"Total exames: {total_exames}, Contraste: {total_contraste}, Sem contraste: {total_sem_contraste}", flush=True)
print(f"Séries válidas: {series_validas}, Sem série válida: {sem_serie_valida}", flush=True)
print(f"Arquivo salvo em: {features_path}", flush=True)
