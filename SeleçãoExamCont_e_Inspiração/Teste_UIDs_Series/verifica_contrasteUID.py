import os
import pydicom
import pandas as pd
from unidecode import unidecode

# ---------------- CONFIGURAÇÕES ----------------
root_folder = r"/Storage/jerogalsky-2024"
planilha_path = r"/home/jerogalsky/IC_PAOLA/patIDStudy_contrast_VPaola.csv"

# ---------------- LEITURA PLANILHA ----------------
planilha = pd.read_csv(planilha_path, dtype=str)
planilha["Contraste"] = planilha["Contraste"].astype(int)

uids_com_contraste = set(planilha.loc[planilha["Contraste"] == 1, "UID dicom"])

print(f"✅ Total de exames com contraste na planilha: {len(uids_com_contraste)}")

# ---------------- VARIÁVEIS AUXILIARES ----------------
encontrados_contraste = set()
nao_encontrados = []
descricoes_encontradas = {}

# ---------------- PERCORRE PASTAS ----------------
for paciente in os.listdir(root_folder):
    paciente_path = os.path.join(root_folder, paciente)
    if not os.path.isdir(paciente_path):
        continue

    for estudo_uid in os.listdir(paciente_path):
        if estudo_uid not in uids_com_contraste:
            continue  # só analisamos os com contraste esperados

        estudo_path = os.path.join(paciente_path, estudo_uid)
        if not os.path.isdir(estudo_path):
            continue

        series_descricoes = []
        for serie_uid in os.listdir(estudo_path):
            serie_path = os.path.join(estudo_path, serie_uid)
            if not os.path.isdir(serie_path):
                continue
            dicoms = [f for f in os.listdir(serie_path) if f.lower().endswith(".dcm")]
            if not dicoms:
                continue
            try:
                ds = pydicom.dcmread(os.path.join(serie_path, dicoms[0]), stop_before_pixels=True)
                desc = unidecode(getattr(ds, "SeriesDescription", "ND")).upper()
                series_descricoes.append(desc)
            except Exception:
                continue

        if not series_descricoes:
            nao_encontrados.append(estudo_uid)
            continue

        descricoes_encontradas[estudo_uid] = series_descricoes

        # critério de contraste (flexível)
        if any(k in d for d in series_descricoes for k in ["CONTRASTE", "CE", "C/C", "ANGIO", "VENOSO"]):
            encontrados_contraste.add(estudo_uid)

# ---------------- RESULTADOS ----------------
print("\n📊 RESULTADOS")
print(f"→ Exames com contraste esperados: {len(uids_com_contraste)}")
print(f"→ Exames com contraste detectados: {len(encontrados_contraste)}")

faltando = uids_com_contraste - encontrados_contraste
if faltando:
    print(f"\n⚠️ Exames que deveriam ter contraste mas não foram detectados ({len(faltando)}):")
    for uid in list(faltando)[:20]:
        print(f"   - {uid}")
        if uid in descricoes_encontradas:
            for desc in descricoes_encontradas[uid]:
                print(f"       Série: {desc}")

print("\n✅ Diagnóstico concluído!")
