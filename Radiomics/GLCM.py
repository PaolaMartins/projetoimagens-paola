import os
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import SimpleITK as sitk
from radiomics import featureextractor

#36136 nro processo
# ===================================================================
# CONFIGURAÇOES
root_folder = r"C:\Users\Home\Documents\USP2024\IC\Pacientes"

planilha_contraste = r"C:\Users\Home\Documents\USP2024\IC\Codigos\SeleçãoExamCont_e_Inspiração\Teste_UIDs_Series\ct_torax_contraste_NumeroExames.csv"
planilha_sem_contraste = r"C:\Users\Home\Documents\USP2024\IC\Codigos\SeleçãoExamCont_e_Inspiração\Teste_UIDs_Series\ct_torax_SC_NumeroExames.csv"

features_path =r"C:\Users\Home\Documents\USP2024\IC\Codigos\MachineLearning\Radiomics\featuresGLCM_teste.csv"

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

# Inicializa o extrator (só precisa ser feito 1 vez)
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('glcm')  # ativa apenas GLCM
extractor.enableImageTypeByName('Original')  # usa a imagem original

def extract_glcm_features(volume, patient_id, series_uid, series_desc, has_contrast):
    """
    Recebe `volume` que pode ser um SimpleITK.Image ou um numpy.ndarray (z,y,x ou x,y,z).
    Retorna dicionário com features GLCM + metadados.
    """
    import SimpleITK as sitk
    import numpy as np

    # 1) Garantir que temos um SimpleITK.Image
    if isinstance(volume, sitk.Image):
        sitk_image = volume
    else:
        # Se for numpy, assumir que está em (slices, rows, cols) ou (rows, cols, slices)
        arr = np.asarray(volume)
        if arr.ndim != 3:
            raise ValueError(f"Volume numpy deve ser 3D, shape atual: {arr.shape}")
        # Heurística simples: se a primeira dimensão não é igual ao número de cortes, transpor
        if arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (2, 0, 1))
        sitk_image = sitk.GetImageFromArray(arr)

    # 2) Criar máscara preenchida com 1 com mesma geometria (size, spacing, origin, direction)
    mask = sitk.Image(sitk_image.GetSize(), sitk.sitkUInt8)
    mask = mask + 1  # todos os voxels = 1
    mask.CopyInformation(sitk_image)  # alinhamento espacial

    # 3) Extrair features
    try:
        result = extractor.execute(sitk_image, mask)
    except Exception as e:
        print(f"   ❌ Erro na extração pyradiomics para {series_uid}: {e}", flush=True)
        return None

    # 4) Filtrar somente as features GLCM
    glcm_feats = {}
    for k, v in result.items():
        if 'glcm' in k.lower():
            try:
                glcm_feats[k] = float(v)
            except:
                pass

    # 5) Adicionar metadados
    glcm_feats.update({
        "PatientID": patient_id,
        "SeriesUID": series_uid,
        "SeriesDescription": series_desc,
        "HasContrast": has_contrast
    })

    return glcm_feats



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
        feats = extract_glcm_features(
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
