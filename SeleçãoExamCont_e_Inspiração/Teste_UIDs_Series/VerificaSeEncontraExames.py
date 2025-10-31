import os
import pydicom
import pandas as pd

# ------------------------- CONFIGURAÇÕES ----------------------------
root_folder = r"/Storage/jerogalsky-2024"

planilha_contraste = r"/home/jerogalsky/IC_PAOLA/teste/ct_torax_contraste_NumeroExames.csv"
planilha_sem_contraste = r"/home/jerogalsky/IC_PAOLA/teste/ct_torax_SC_NumeroExames.csv"

# -------------------- CARREGA LISTAS DE UIDs --------------------------
df1 = pd.read_csv("planilha_contraste", sep=";", dtype=str)
uids_contraste = df1["_id"].tolist()
df2 = pd.read_csv("planilha_sem_contraste", sep=";", dtype=str)
uids_sem_contraste = df2["_id"].tolist()

# -------------------- VARIÁVEIS DE CONTAGEM --------------------------
total_exames = 0
total_contraste = 0
total_sem_contraste = 0
sem_serie_valida = 0
series_validas = 0

resumo_exames = []

# -------------------- LOOP DE VERIFICAÇÃO -----------------------------
for paciente in os.listdir(root_folder):
    paciente_path = os.path.join(root_folder, paciente)
    if not os.path.isdir(paciente_path):
        continue

    for estudo_uid in os.listdir(paciente_path):
        estudo_path = os.path.join(paciente_path, estudo_uid)
        if not os.path.isdir(estudo_path):
            continue

        # Verifica se o estudo está nas planilhas
        if estudo_uid in uids_contraste:
            contraste_flag = 1
        elif estudo_uid in uids_sem_contraste:
            contraste_flag = 0
        else:
            continue  # ignora se não estiver em nenhuma planilha

        total_exames += 1
        if contraste_flag == 1:
            total_contraste += 1
        else:
            total_sem_contraste += 1

        best_series = None
        max_slices = -1

        # Verifica todas as séries dentro do exame
        for serie_uid in os.listdir(estudo_path):
            serie_path = os.path.join(estudo_path, serie_uid)
            if not os.path.isdir(serie_path):
                continue

            dicoms = [f for f in os.listdir(serie_path) if f.endswith(".dcm")]
            if not dicoms:
                continue

            try:
                ds = pydicom.dcmread(os.path.join(serie_path, dicoms[0]), stop_before_pixels=True)
                rows = getattr(ds, "Rows", None)
                cols = getattr(ds, "Columns", None)
                n_slices = len(dicoms)
                desc = getattr(ds, "SeriesDescription", "ND")
            except Exception:
                continue

            # Filtra por resolução 512x512
            if rows == 512 and cols == 512:
                if n_slices > max_slices:
                    max_slices = n_slices
                    best_series = {
                        "PatientID": paciente,
                        "StudyUID": estudo_uid,
                        "SeriesUID": serie_uid,
                        "SeriesDescription": desc,
                        "NumSlices": n_slices,
                        "Resolution": f"{rows}x{cols}",
                        "HasContrast": contraste_flag
                    }

        if best_series:
            series_validas += 1
            resumo_exames.append(best_series)
        else:
            sem_serie_valida += 1

# -------------------- RELATÓRIO FINAL -----------------------------
print("\n📊 RELATÓRIO DE TESTE")
print(f"Total de exames nas planilhas: {len(uids_contraste) + len(uids_sem_contraste)}")
print(f"Total de exames encontrados no servidor: {total_exames}")
print(f" ├── Com contraste: {total_contraste}")
print(f" ├── Sem contraste: {total_sem_contraste}")
print(f"Exames com série 512x512 válida: {series_validas}")
print(f"Exames sem série válida: {sem_serie_valida}")

# -------------------- EXPORTA RESULTADOS -----------------------------
df_resumo = pd.DataFrame(resumo_exames)
df_resumo.to_csv("/home/jerogalsky/IC_PAOLA/teste/exames_validos_teste.csv", index=False)

print("\n✅ Teste concluído!")
print("Arquivo salvo como 'exames_validos_teste.csv'.")
print("Contém apenas a série com mais cortes (512x512) de cada exame encontrado.")
