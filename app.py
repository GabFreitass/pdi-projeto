# -*- coding: utf-8 -*-
# =============================================================================
# App de demonstração — Detecção de objetos (YOLOv8n x SSD-MobileNet)
# Trabalho de Processamento Digital de Imagens.
#
# O app APENAS carrega pesos já treinados e roda inferência (não treina nada).
# Combinação escolhida na sidebar (Objeto x Arquitetura) decide qual dos 4
# pesos em models/ será carregado.
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORTANTE: forçar modo offline do Ultralytics ANTES de importar a biblioteca.
# Assim a demo nunca depende de internet (sem checagem de versão no PyPI e sem
# download de fontes). Precisa vir antes de "from ultralytics import ...".
# -----------------------------------------------------------------------------
import os
os.environ["YOLO_OFFLINE"] = "true"

import cv2
import numpy as np
import streamlit as st
import torch
import torchvision
import torchvision.transforms.functional as TF
from functools import partial
from PIL import Image

from ultralytics import YOLO
try:
    # Desliga analytics/HUB do Ultralytics (best-effort; ignora se a versão não suportar)
    from ultralytics import settings as _yolo_settings
    _yolo_settings.update({"sync": False})
except Exception:
    pass

# streamlit-webrtc é OPCIONAL: habilita a webcam ao vivo (quadro a quadro).
# Se a lib (ou av/aiortc) não estiver instalada, WEBRTC_OK=False e o modo Webcam
# cai automaticamente para st.camera_input (foto). O app nunca quebra por isso.
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    WEBRTC_OK = True
except Exception:
    WEBRTC_OK = False


# -----------------------------------------------------------------------------
# Constantes e configuração
# -----------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
PASTA_MODELOS = os.path.join(BASE, "models")

# Mapa (Objeto, Arquitetura) -> nome do arquivo de peso dentro de models/
MODELOS = {
    ("Maçãs", "YOLOv8n"):          "maca_yolo.pt",
    ("Garrafas", "YOLOv8n"):       "garrafa_yolo.pt",
    ("Maçãs", "SSD-MobileNet"):    "maca_mobilenet.pth",
    ("Garrafas", "SSD-MobileNet"): "garrafa_mobilenet.pth",
}

# Nome de classe exibido conforme o objeto escolhido (cada peso é de 1 objeto só)
NOMES_CLASSE = {"Maçãs": "Maçã", "Garrafas": "Garrafa"}

# Cores das caixas por classe, em BGR (paleta permitida, distintas e legíveis)
CORES = {
    "Maçã":    (0, 91, 255),   # laranja #FF5B00
    "Garrafa": (180, 88, 60),  # azul   #3C58B4
}

CONF_DEFAULT = 0.60   # limiar de confiança usado na avaliação do trabalho
IOU_NMS = 0.45        # IoU do NMS aplicado ao SSD-MobileNet


# -----------------------------------------------------------------------------
# Carregamento dos modelos (cacheado: cada peso é carregado uma única vez)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo YOLOv8n...")
def carregar_yolo(caminho):
    """Carrega um modelo YOLOv8n (.pt) da Ultralytics a partir de caminho absoluto."""
    return YOLO(caminho)


def _construir_ssd(num_classes=2):
    """Constrói a arquitetura SSDLite320-MobileNetV3-Large que os checkpoints usam.

    Por que NÃO usamos ssdlite320_mobilenet_v3_large() direto:
    o construtor do torchvision acopla o "reduced tail" ao backbone pré-treinado
    (`reduce_tail = weights_backbone is None`). Ou seja:
      - weights_backbone=None  -> reduced tail (backbone com metade dos canais) + offline
      - weights_backbone=<pesos> -> tail COMPLETO, porém baixa o backbone da internet
    Os nossos .pth foram treinados com o tail COMPLETO (chamada padrão, que baixava
    o backbone no treino). Para casar a arquitetura SEM depender de internet na demo,
    reconstruímos o modelo manualmente com `reduced_tail=False` e `weights=None`
    (sem download). O state_dict treinado sobrescreve todos os pesos em seguida.
    """
    # imports internos do torchvision feitos aqui (lazy) para não derrubar o app
    # inteiro caso uma versão diferente mude a API — o erro vira st.error amigável.
    from torch import nn
    from torchvision.models import mobilenet_v3_large
    from torchvision.models.detection import _utils as det_utils
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    from torchvision.models.detection.ssd import SSD
    from torchvision.models.detection.ssdlite import SSDLiteHead, _mobilenet_extractor

    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    # tail COMPLETO (reduced_tail=False) e sem pesos pré-treinados (weights=None -> sem rede)
    backbone = mobilenet_v3_large(weights=None, norm_layer=norm_layer, reduced_tail=False)
    backbone = _mobilenet_extractor(backbone, 6, norm_layer)

    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    # mesmos defaults do construtor oficial do ssdlite320
    defaults = {
        "score_thresh": 0.001, "nms_thresh": 0.55, "detections_per_img": 300,
        "topk_candidates": 300, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    }
    return SSD(
        backbone, anchor_generator, size, num_classes,
        head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer), **defaults,
    )


@st.cache_resource(show_spinner="Carregando modelo SSD-MobileNet...")
def carregar_ssd(caminho):
    """Carrega SSD-MobileNet (.pth): state_dict puro sobre a arquitetura de tail completo."""
    modelo = _construir_ssd(num_classes=2)
    estado = torch.load(caminho, map_location="cpu", weights_only=True)
    modelo.load_state_dict(estado)  # strict=True: tem que casar 100% com a arquitetura
    modelo.eval()
    return modelo


def obter_modelo(objeto, arquitetura):
    """Resolve o peso da combinação, valida existência e carrega (cacheado).

    Retorna (modelo, erro): em caso de problema, modelo=None e erro é uma
    mensagem amigável (nunca deixamos stack trace cru aparecer na tela).
    """
    nome = MODELOS[(objeto, arquitetura)]
    caminho = os.path.join(PASTA_MODELOS, nome)

    if not os.path.exists(caminho):
        return None, (
            f"Arquivo de pesos não encontrado: **models/{nome}**.\n\n"
            f"Coloque o peso treinado nessa pasta com esse nome exato e tente de novo."
        )

    try:
        if arquitetura == "YOLOv8n":
            return carregar_yolo(caminho), None
        return carregar_ssd(caminho), None
    except Exception as e:
        return None, (
            f"Não foi possível carregar **models/{nome}**. "
            f"Confira se o arquivo corresponde à arquitetura **{arquitetura}**.\n\n"
            f"Detalhe técnico: {type(e).__name__}: {e}"
        )


# -----------------------------------------------------------------------------
# Pré-processamento opcional (aplicado antes da inferência). Entrada/saída BGR.
# -----------------------------------------------------------------------------
def preprocess(img_bgr):
    """Realce leve com OpenCV: CLAHE no canal de luminância + blur gaussiano."""
    # 1) CLAHE no canal L (luminância) do espaço LAB -> melhora o contraste local
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # 2) Gaussian blur leve (kernel 3x3) -> suaviza ruído fino
    img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    return img_bgr


# -----------------------------------------------------------------------------
# Inferência — cada função devolve uma lista uniforme de detecções:
#   {"classe": str, "conf": float, "box": [x1, y1, x2, y2]}
# -----------------------------------------------------------------------------
def detectar_yolo(modelo, img_bgr, conf, nome_classe):
    """Inferência YOLOv8n. O NMS já é interno do YOLO."""
    resultados = modelo(img_bgr, conf=conf, verbose=False)
    deteccoes = []
    if not resultados or resultados[0].boxes is None:
        return deteccoes
    boxes = resultados[0].boxes
    for xyxy, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = xyxy.tolist()
        deteccoes.append({
            "classe": nome_classe,
            "conf": float(c),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        })
    return deteccoes


def detectar_ssd(modelo, img_bgr, conf, nome_classe):
    """Inferência SSD-MobileNet. Filtra por score e classe=1, aplica NMS (IoU 0.45)."""
    # O modelo recebe uma LISTA de tensores [C,H,W] em float [0,1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(Image.fromarray(img_rgb))  # já normaliza para [0,1]
    with torch.no_grad():
        saida = modelo([tensor])[0]  # dict: 'boxes', 'labels', 'scores'

    boxes, scores, labels = saida["boxes"], saida["scores"], saida["labels"]

    # mantém apenas o objeto (label==1; 0 é o fundo) acima do limiar
    keep = (scores >= conf) & (labels == 1)
    boxes, scores = boxes[keep], scores[keep]

    deteccoes = []
    if boxes.numel() == 0:
        return deteccoes

    # NMS explícito (diferente do YOLO, o SSD não faz isso por nós aqui)
    idx = torchvision.ops.nms(boxes, scores, IOU_NMS)
    for i in idx:
        x1, y1, x2, y2 = boxes[i].tolist()
        deteccoes.append({
            "classe": nome_classe,
            "conf": float(scores[i]),
            "box": [int(x1), int(y1), int(x2), int(y2)],
        })
    return deteccoes


# -----------------------------------------------------------------------------
# Desenho das caixas — função única reusada por imagem e webcam
# -----------------------------------------------------------------------------
def draw_boxes(img_bgr, deteccoes):
    """Desenha retângulo + rótulo 'classe conf%' para cada detecção. Retorna BGR."""
    img = img_bgr.copy()
    for d in deteccoes:
        x1, y1, x2, y2 = d["box"]
        cor = CORES.get(d["classe"], (0, 91, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)

        rotulo = f'{d["classe"]} {d["conf"] * 100:.0f}%'
        (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_top = max(y1 - th - 8, 0)
        # retângulo de fundo para o texto ficar sempre legível
        cv2.rectangle(img, (x1, y_top), (x1 + tw + 6, y_top + th + 8), cor, -1)
        cv2.putText(img, rotulo, (x1 + 3, y_top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# -----------------------------------------------------------------------------
# Utilitário — decodifica bytes (upload OU câmera) em imagem BGR (np.ndarray)
# -----------------------------------------------------------------------------
def bytes_para_bgr(buf):
    """cv2.imdecode entrega 3 canais BGR direto, evitando problemas de RGBA/paleta."""
    dados = np.frombuffer(buf.getvalue(), np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def processar_estatico(img_bgr, objeto, arquitetura, conf, usar_preproc):
    """Detecção numa imagem ÚNICA (upload ou foto) + exibição de imagem e tabela.

    Reaproveitada pelo modo Imagem e pelo fallback de foto da Webcam. Mostra erros
    amigáveis via st.error — nunca deixa stack trace na tela.
    """
    # pré-processamento opcional (fácil de ligar/desligar pelo checkbox)
    if usar_preproc:
        img_bgr = preprocess(img_bgr)

    # carrega o modelo da combinação escolhida (cacheado)
    modelo, erro = obter_modelo(objeto, arquitetura)
    if erro:
        st.error(erro)
        return

    nome_classe = NOMES_CLASSE[objeto]
    try:
        if arquitetura == "YOLOv8n":
            deteccoes = detectar_yolo(modelo, img_bgr, conf, nome_classe)
        else:
            deteccoes = detectar_ssd(modelo, img_bgr, conf, nome_classe)
    except Exception as e:
        st.error(f"Falha na inferência: {type(e).__name__}: {e}")
        return

    # desenha as caixas e mostra a imagem (converter BGR->RGB só aqui)
    img_saida = draw_boxes(img_bgr, deteccoes)
    st.image(
        cv2.cvtColor(img_saida, cv2.COLOR_BGR2RGB),
        caption=f"{objeto} · {arquitetura} · confiança ≥ {conf:.2f}",
        width='stretch',
    )

    if deteccoes:
        st.subheader(f"Detecções ({len(deteccoes)})")
        tabela = [
            {"#": i + 1, "Classe": d["classe"], "Confiança": f'{d["conf"] * 100:.1f}%'}
            for i, d in enumerate(deteccoes)
        ]
        st.dataframe(tabela, width='stretch', hide_index=True)
    else:
        st.warning("Nenhuma detecção acima do limiar de confiança.")


# =============================================================================
# Interface
# =============================================================================
st.set_page_config(page_title="Detecção de Objetos — PDI", layout="wide")

# Cabeçalho (framing acadêmico neutro, dentro da paleta permitida)
st.markdown(
    """
    <div style="border-left: 6px solid #FF5B00; padding: 0.2rem 1rem; margin-bottom: 1rem;">
      <h1 style="color:#05244F; margin-bottom:0.2rem;">
        Detecção de Objetos — YOLOv8n × SSD-MobileNet
      </h1>
      <p style="color:#273A76; margin:0;">
        Processamento Digital de Imagens · Comparação de arquiteturas (maçãs e garrafas)
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: controles ------------------------------------------------------
with st.sidebar:
    st.header("Controles")
    objeto = st.radio("Objeto", ["Maçãs", "Garrafas"])
    arquitetura = st.radio("Arquitetura", ["YOLOv8n", "SSD-MobileNet"])
    entrada = st.radio("Entrada", ["Imagem", "Webcam"])
    usar_preproc = st.checkbox("Pré-processamento (CLAHE + blur)", value=False)
    conf = st.slider("Confiança", 0.0, 1.0, CONF_DEFAULT, 0.01)
    executar = st.button("Executar", type="primary", width='stretch')

# --- Área principal: entrada + resultado -------------------------------------
if entrada == "Imagem":
    # Modo Imagem: upload + Executar -> detecção estática.
    buf = st.file_uploader("Envie uma imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if executar:
        if buf is None:
            st.warning("Forneça uma entrada primeiro: envie uma imagem.")
        else:
            img_bgr = bytes_para_bgr(buf)
            if img_bgr is None:
                st.error("Não foi possível ler a imagem. Tente outro arquivo.")
            else:
                processar_estatico(img_bgr, objeto, arquitetura, conf, usar_preproc)

else:  # entrada == "Webcam"
    # Modo Webcam: preferencial = detecção AO VIVO (streamlit-webrtc); se a lib não
    # estiver disponível ou o stream não iniciar, cai para foto (st.camera_input).
    modelo, erro = obter_modelo(objeto, arquitetura)   # cacheado; mesmo dos outros modos
    nome_classe = NOMES_CLASSE[objeto]

    if erro:
        # sem o peso não dá para detectar (nem ao vivo, nem por foto)
        st.error(erro)
    else:
        usar_webrtc = WEBRTC_OK
        if WEBRTC_OK:
            # O callback é recriado a cada rerun, capturando os valores ATUAIS dos
            # controles (objeto/arquitetura/conf/pré-proc) — assim o slider e os
            # radios atualizam a detecção ao vivo. Roda em OUTRA thread: nada de
            # st.* aqui dentro, e tudo em try/except para nunca derrubar o stream.
            def _cb(frame):
                img = frame.to_ndarray(format="bgr24")
                try:
                    # downscale p/ manter uma taxa de quadros utilizável em CPU
                    h, w = img.shape[:2]
                    if w > 640:
                        img = cv2.resize(img, (640, int(h * 640 / w)))
                    if usar_preproc:
                        img = preprocess(img)
                    if arquitetura == "YOLOv8n":
                        dets = detectar_yolo(modelo, img, conf, nome_classe)
                    else:
                        dets = detectar_ssd(modelo, img, conf, nome_classe)
                    img = draw_boxes(img, dets)
                except Exception:
                    pass  # em qualquer erro, devolve o frame como veio
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            try:
                webrtc_streamer(
                    key="webcam",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=_cb,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                st.caption(
                    "Detecção ao vivo — as caixas seguem o objeto em tempo real. "
                )
            except Exception:
                usar_webrtc = False
                st.info("Modo ao vivo indisponível — usando captura de foto como alternativa.")

        if not usar_webrtc:
            # Fallback: foto + Executar, reaproveitando o mesmo fluxo do modo Imagem.
            if not WEBRTC_OK:
                st.info("streamlit-webrtc não instalado — usando captura de foto (st.camera_input).")
            buf = st.camera_input("Capture um frame com a webcam")
            if executar:
                if buf is None:
                    st.warning("Forneça uma entrada primeiro: tire uma foto com a webcam.")
                else:
                    img_bgr = bytes_para_bgr(buf)
                    if img_bgr is None:
                        st.error("Não foi possível ler a imagem. Tente novamente.")
                    else:
                        processar_estatico(img_bgr, objeto, arquitetura, conf, usar_preproc)
