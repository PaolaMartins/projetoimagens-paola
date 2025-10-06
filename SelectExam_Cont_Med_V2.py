import os
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

# ------------------------- CONFIGURAÇÕES ----------------------------
root_folder = r"C:\Users\Home\Documents\USP2024\IC\Pacientes"
planilha_path = r"C:\Users\Home\Documents\USP2024\IC\Codigos\SeleçãoExamCont_e_Inspiração\patIDStudy_contrast_VPaola.csv"
features_path = r"C:\Users\Home\Documents\USP2024\IC\Codigos\SeleçãoExamCont_e_Inspiração"

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
                        "Series Description": getattr(ds, "SeriesDescription", "ND"),
                        "Series Instance UID": ds.SeriesInstanceUID,
                        "Number of Slices": len(dicoms)
                    })
                except Exception as e:
                    print(f"Erro ao ler {serie_path}: {e}")
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


def select_series_by_rules(series_list, contraste_flag):
    """Seleciona a série com base nas regras:
    - Se contraste == 1 → pega a série com mais de 300 cortes
    - Se contraste == 0 → tenta MEDIASTINO, senão MED INSPIRAÇÃO, senão INSPIRAÇÃO
    """
    for s in series_list:
        s["desc_upper"] = s["Series Description"].upper()

    if contraste_flag == 1:
        candidatas = [s for s in series_list if s["Number of Slices"] > 300]
    else:
        candidatas = []
        for keyword in ["MEDIASTINO", "MED INSPIRAÇÃO ", "MED INSPIRACAO Body 1.0","MEDIASTINO Body 1.0", "INSPIRAÇÃO","INSPIRACAO", "MED"]:
            candidatas = [s for s in series_list if keyword in s["desc_upper"]]
            if candidatas:
                break

    if not candidatas:
        return None

    return max(candidatas, key=lambda s: s["Number of Slices"])


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


def extract_histogram_features(volume_array, patient_id, series_id, series_description, has_contrast):
    array = volume_array.flatten()
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


# -------------------- EXECUÇÃO PRINCIPAL ----------------------------

# Lê a planilha de controle
planilha = pd.read_csv(planilha_path, dtype=str)
planilha["Contraste"] = planilha["Contraste"].astype(int)

# Extrai metadados dos pacientes
pacientes_data = extract_studies_metadata(root_folder)

hist_folder = os.path.join(root_folder, "Histogramas")
os.makedirs(hist_folder, exist_ok=True)

features_list = []

for patient_id, studies in pacientes_data.items():
    print(f"\n📂 Paciente: {patient_id}")

    for study in studies:
        study_uid = study["Study Instance UID"]

        # Verifica se UID está na planilha
        row = planilha.loc[planilha["UID dicom"] == study_uid]
        if row.empty:
            continue  # UID não listado na planilha

        contraste_flag = int(row["Contraste"].iloc[0])
        selected_series = select_series_by_rules(study["Series"], contraste_flag)

        if selected_series is None:
            print(f"   !-Nenhuma série elegível encontrada para estudo {study_uid}")
            continue

        series_uid = selected_series["Series Instance UID"]
        series_path = os.path.join(root_folder, patient_id, study_uid, series_uid)
        desc = selected_series["Series Description"]

        print(f"   -Série selecionada: {desc} ({selected_series['Number of Slices']} cortes)")

        # Carrega volume e gera histograma
        volume = loadSeries(series_path)
        array = sitk.GetArrayFromImage(volume)

        # Remove pixels de valor zero (fundo)
        array = array[array != 0]

        features = extract_histogram_features(array, patient_id, series_uid, desc, contraste_flag)
        features_list.append(features)

        # Gera histograma apenas dos pixels válidos
        plt.figure(figsize=(8, 4))
        plt.hist(array, bins=100, color='steelblue', edgecolor='black', log=True)
        plt.title(f"Histograma - Paciente {patient_id}\n{desc}")
        plt.xlabel("Intensidade (1–255)")
        plt.ylabel("Frequência")
        plt.tight_layout()

        # Remove caracteres inválidos do nome do arquivo
        safe_desc = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in desc)
        filename = f"{patient_id}_{safe_desc}_histograma.png"

        plt.savefig(os.path.join(hist_folder, filename))
        plt.close()
        print(f"     - Histograma salvo: {filename}")


# Salva tabela de features
df_features = pd.DataFrame(features_list)
output_csv = os.path.join(features_path, "features_histogramas.csv")
df_features.to_csv(output_csv, index=False)
print("\nExtração concluída! Planilha salva como 'features_histogramas.csv'.")


