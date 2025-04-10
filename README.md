# Projeto de Diferenciação de Imagens de TC de Tórax com e sem Contraste

Este projeto tem como objetivo o desenvolvimento de um algoritmo baseado em inteligência artificial (IA) para a diferenciação automatizada de imagens de tomografia computadorizada (TC) de tórax com e sem contraste.

## 🧠 Sobre o Projeto

A proposta é estudar e implementar um modelo de IA capaz de distinguir exames de TC de tórax com e sem contraste de forma precisa. O projeto envolve:

- Treinamento e validação do algoritmo com base em um banco de dados previamente existente.
- Comparação dos resultados do modelo com as informações contidas nos cabeçalhos DICOM das imagens.
- Validação de concordância entre os resultados do modelo e das tags DICOM.
- Encaminhamento do exame para inspeção visual em caso de divergência na comparação.


## 🗂️ Sobre a Base de Dados

A base utilizada no projeto contém exames de tomografia computadorizada (TC) de **199 pacientes**, com imagens representando cortes axiais dos **pulmões e estruturas torácicas**. Esses exames foram adquiridos em sua maioria no setor de Radiologia do HCFMRP-USP, utilizando **tomógrafos multidetectores de 16 ou 80 canais** (Philips Brillance Big Bore e Toshiba Aquilion Prime).

Cada exame é composto por aproximadamente **300 cortes por série**, reconstruídos volumetricamente com espessura de **1 mm**, resolução espacial de **512×512 ou 768×768 pixels**, e aplicados filtros padrão (janela mediastinal) e filtro duro (janela pulmonar e óssea).

Para a inspeção da região mediastinal, será utilizada uma **janela mediastinal** com largura de **400 HU** e nível central de **40 HU**, o que favorece a visualização de órgãos, vasos e tecidos moles na região torácica.

## ⚙️ Tecnologias e Ferramentas

- **Linguagem:** Python  
- **IA:** Aprendizado de Máquina e Aprendizado Profundo  
- **Banco de Dados:** MongoDB  
- **Softwares auxiliares:** 3DSlicer  
- **IDE:** Visual Studio Code

## Status do Projeto

🟡 **Em estudo** – Atualmente está sendo feita a análise inicial da base de dados e o levantamento estatístico das informações.

## Discente

**Paola Martins**  
🔗 GitHub: [@PaolaMartins](https://github.com/PaolaMartins)

---


