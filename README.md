# Detecção de Objetos — YOLOv8n × SSD-MobileNet

App **Streamlit** para demonstração (trabalho de Processamento Digital de Imagens).
Compara duas arquiteturas — **YOLOv8n** (Ultralytics) e **SSD-MobileNet**
(torchvision) — na detecção de **maçãs** e **garrafas**. O app apenas **carrega
pesos já treinados e roda inferência** — não treina nada.

Escolha na barra lateral o **objeto**, a **arquitetura** e a **entrada** (imagem ou
webcam), ajuste o limiar de **confiança** e clique em **Executar** para ver as
caixas e a tabela de detecções.

## Como rodar

```bash
# 1) Criar e ativar o ambiente virtual
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# 2) Instalar as dependências
pip install -r requirements.txt

# 3) Rodar o app
streamlit run app.py
```

O app abre no navegador (normalmente `http://localhost:8501`). O navegador pedirá
permissão de acesso à câmera na primeira vez.

### Webcam ao vivo (streamlit-webrtc)
O modo **Webcam** faz detecção **ao vivo, quadro a quadro** (as caixas seguem o
objeto em tempo real) via `streamlit-webrtc`. Requisitos e comportamento:

- **Contexto seguro**: a câmera do navegador só funciona em contexto seguro —
  `http://localhost` é aceito. Acessando por IP de outra máquina/LAN, o navegador
  exige **HTTPS** (use um proxy TLS ou o Streamlit Community Cloud).
- **Fallback automático**: se `streamlit-webrtc` não estiver instalado ou o stream
  ao vivo não iniciar, o app cai sozinho para **captura de foto** (`st.camera_input`)
  — nunca quebra. `streamlit-webrtc` traz `av`/`aiortc` como wheels (sem compilador).

## Modelos (pasta `models/`)

Coloque os **4 arquivos de pesos** na pasta `models/`, com estes nomes **exatos**
(a combinação Objeto × Arquitetura decide qual é carregado):

| Objeto   | Arquitetura   | Arquivo em `models/`      |
|----------|---------------|---------------------------|
| Maçãs    | YOLOv8n       | `maca_yolo.pt`            |
| Garrafas | YOLOv8n       | `garrafa_yolo.pt`         |
| Maçãs    | SSD-MobileNet | `maca_mobilenet.pth`      |
| Garrafas | SSD-MobileNet | `garrafa_mobilenet.pth`   |

- **YOLOv8n** (`.pt`): modelo Ultralytics padrão.
- **SSD-MobileNet** (`.pth`): `state_dict` puro treinado sobre
  `ssdlite320_mobilenet_v3_large` com `num_classes=2` (1 classe + fundo).

Os pesos **não são versionados** (ver `.gitignore`); adicione-os manualmente.
Se um peso estiver faltando ou não corresponder à arquitetura, o app mostra uma
mensagem de erro amigável — sem stack trace.

## Observações

- Roda **100% offline** (nenhum download é disparado na inferência).
- **Pré-processamento** opcional (CLAHE + blur leve) via checkbox na barra lateral.
- Limiar de confiança padrão: **0.60**; NMS do SSD com IoU **0.45**.
