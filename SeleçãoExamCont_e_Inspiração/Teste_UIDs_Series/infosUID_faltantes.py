import os
import pydicom
import pandas as pd

# ---------- CONFIGURAÇÃO ----------
root_folder = r"/Storage/jerogalsky-2024"  # raiz com pastas de pacientes
planilha_path = r"/home/jerogalsky/IC_PAOLA/patIDStudy_contrast_VPaola.csv"

# Lê planilha
planilha = pd.read_csv(planilha_path, dtype=str)
planilha["Contraste"] = planilha["Contraste"].astype(int)

# Filtra apenas os UIDs com contraste
uids_com_contraste = planilha.loc[planilha["Contraste"] == 1, "UID dicom"].tolist()

# ---------- FUNÇÃO PARA EXPLORAR PASTAS ----------
def procurar_series_por_uid(root_folder, uid_list):
    resultados = {}  # armazenará info: {UID: [{serie_desc, nro_cortes, serie_uid}, ...]}
    
    for paciente in os.listdir(root_folder):
        paciente_path = os.path.join(root_folder, paciente)
        if not os.path.isdir(paciente_path):
            continue
        
        for estudo_uid in os.listdir(paciente_path):
            if estudo_uid not in uid_list:
                continue  # só nos UIDs que queremos
            estudo_path = os.path.join(paciente_path, estudo_uid)
            if not os.path.isdir(estudo_path):
                continue
            
            series_list = []
            for serie_uid in os.listdir(estudo_path):
                serie_path = os.path.join(estudo_path, serie_uid)
                if not os.path.isdir(serie_path):
                    continue
                
                dicoms = [f for f in os.listdir(serie_path) if f.lower().endswith(".dcm")]
                if not dicoms:
                    continue
                
                try:
                    ds = pydicom.dcmread(os.path.join(serie_path, dicoms[0]), stop_before_pixels=True)
                    series_list.append({
                        "Series Description": getattr(ds, "SeriesDescription", "ND"),
                        "Series UID": getattr(ds, "SeriesInstanceUID", serie_uid),
                        "Number of Slices": len(dicoms)
                    })
                except Exception as e:
                    print(f"Erro ao ler {serie_path}: {e}")
            
            resultados[estudo_uid] = series_list
    return resultados

# ---------- EXECUÇÃO ----------
resultados = procurar_series_por_uid(root_folder, uids_com_contraste)

# Mostra resultados
for uid, series in resultados.items():
    print(f"\nUID: {uid}")
    if not series:
        print("  Nenhuma série encontrada.")
        continue
    for s in series:
        print(f"  Série: {s['Series Description']} | Nº cortes: {s['Number of Slices']} | Serie UID: {s['Series UID']}")
