import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import numpy as np
import pandas as pd
import pydicom
import os

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

# Configuração do PyRadiomics - apenas GLCM
settings = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': 'sitkBSpline',

    'enableCExtensions': True
}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('glcm')

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

def load_dicom_series_to_itk(series_folder):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(series_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def create_full_mask(image):
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    mask = sitk.Add(mask, 1)  # tudo = 1
    return mask

# FUNÇÂO PRINCIPAL 
all_metadata = extract_studies_metadata(root_folder)
results = []

for paciente, studies in all_metadata.items():
    for study in studies:
        for serie in study['Series']:
            serie_path = serie['Folder Path']
            try:
                print(f"\nExtraindo GLCM para: {paciente} | {serie['Series Description']}")
                image = load_dicom_series_to_itk(serie_path)
                mask = create_full_mask(image)
                result = extractor.execute(image, mask)

                # Filtra apenas features GLCM numéricas
                glcm_feats = {k: v for k, v in result.items() if 'glcm' in k.lower() and isinstance(v, (int, float))}

                # Adiciona metadados úteis
                glcm_feats.update({
                    "Paciente": paciente,
                    "StudyUID": study["Study Instance UID"],
                    "SeriesUID": serie["Series Instance UID"],
                    "Descricao": serie["Series Description"]
                })
                results.append(glcm_feats)

            except Exception as e:
                print(f"❌ Erro ao processar {serie_path}: {e}")
                continue

# Salva todas as features em CSV
df_features = pd.DataFrame(results)
df_features.to_csv(features_path, index=False)
print(f"\n✅ Extração concluída. Features salvas em: {features_path}")

