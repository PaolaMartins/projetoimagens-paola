import os
import pandas as pd

# ------------------------- CONFIGURAÇÕES ----------------------------
root_folder = r"/Storage/jerogalsky-2024"
planilha_path = r"/home/jerogalsky/IC_PAOLA/patIDStudy_contrast_VPaola.csv"
modo_teste = True  # não processa imagens
# -------------------------------------------------------------------

# Lê a planilha
planilha = pd.read_csv(planilha_path, dtype=str)
planilha["Contraste"] = planilha["Contraste"].astype(int)

# Coletar todos os UIDs encontrados no servidor
uids_servidor = set()
for paciente in os.listdir(root_folder):
    paciente_path = os.path.join(root_folder, paciente)
    if not os.path.isdir(paciente_path):
        continue

    for study_uid in os.listdir(paciente_path):
        study_path = os.path.join(paciente_path, study_uid)
        if os.path.isdir(study_path):
            uids_servidor.add(study_uid.strip())

# Coletar todos os UIDs da planilha
uids_planilha = set(planilha["UID dicom"].str.strip())

# Interseção entre os dois conjuntos
uids_comuns = uids_servidor.intersection(uids_planilha)

# Ver quantos desses têm contraste == 1
planilha_contraste = planilha.loc[planilha["Contraste"] == 1, "UID dicom"].str.strip()
uids_contraste_planilha = set(planilha_contraste)
uids_contraste_encontrados = uids_contraste_planilha.intersection(uids_servidor)

# -------------------- RESULTADOS --------------------
print("\n📊 RESULTADOS GERAIS 📊")
print(f"Total de UIDs na planilha: {len(uids_planilha)}")
print(f"Total de UIDs no servidor: {len(uids_servidor)}")
print(f"UIDs encontrados em comum: {len(uids_comuns)}")

print("\n💉 COM CONTRASTE:")
print(f" - Total na planilha: {len(uids_contraste_planilha)}")
print(f" - Encontrados no servidor: {len(uids_contraste_encontrados)}")
print(f" - Faltando no servidor: {len(uids_contraste_planilha - uids_servidor)}")

# Se quiser listar os faltantes:
if modo_teste:
    faltantes = sorted(list(uids_contraste_planilha - uids_servidor))
    if faltantes:
        print("\n⚠️ UIDs com contraste que estão NA PLANILHA mas NÃO no servidor:")
        for uid in faltantes:
            print(f"  - {uid}")
    else:
        print("\n✅ Todos os UIDs com contraste da planilha foram encontrados no servidor!")
