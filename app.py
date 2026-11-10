import os
from pathlib import Path
from io import BytesIO
from PIL import Image
import streamlit as st
import pandas as pd
import torch
from dotenv import load_dotenv
from pipeline.preprocessing import prepare_image
from pipeline.prediction import load_h3,predict
from pipeline.gradcam import make_gradcams,overlay
from reporting.report_template import make_report_text
from reporting.pdf_generator import build_pdf

load_dotenv()
st.set_page_config(page_title="RetinaAI | Retinal screening",page_icon="👁️",layout="wide")
st.title("RetinaAI — Multi-condition retinal screening")
st.caption("Research demonstration • H3 hybrid CNN–Transformer • Not a medical diagnosis")
DEFAULT_PATH=Path(__file__).parent/'models'/'best_h3_hybrid.pth'
if not DEFAULT_PATH.exists():
    st.error("Missing models/best_h3_hybrid.pth. Copy your verified H3 checkpoint here (do not rename another model).")
    st.stop()
@st.cache_resource(show_spinner="Loading H3 checkpoint...")
def get_model():
    return load_h3(DEFAULT_PATH,device="cpu")
try:model=get_model()
except Exception as exc:
    st.error(f"Checkpoint loading failed: {exc}");st.stop()

file=st.file_uploader("Upload retinal fundus image",type=['png','jpg','jpeg'])
if file:
    try:
        raw=Image.open(BytesIO(file.getvalue())).convert('RGB')
        x,processed,mask=prepare_image(raw)
        result=predict(model,x,device="cpu")
        cnn_cam,trans_cam=make_gradcams(model,x,result['predicted_index'])
        cnn_overlay=overlay(processed,cnn_cam,mask)
        trans_overlay=overlay(processed,trans_cam,mask)
        a,b,c=st.columns(3)
        with a:st.image(processed,caption="P2 Retinex processed image",use_container_width=True)
        with b:st.image(cnn_overlay,caption="CNN feature Grad-CAM",use_container_width=True)
        with c:st.image(trans_overlay,caption="Transformer token attribution",use_container_width=True)
        st.subheader(f"Predicted category: {result['predicted_class']}")
        st.caption("Scores are uncalibrated softmax outputs, NOT clinical probabilities")
        df=pd.DataFrame({"Class":list(result['probabilities'].keys()),"Model score":list(result['probabilities'].values())})
        st.dataframe(df,hide_index=True,use_container_width=True)
        report=make_report_text(result,file.name)
        summary=None
        if st.checkbox("Add optional LangChain-generated narrative",value=False):
            try:
                from reporting.langchain_report import draft_summary_with_langchain
                summary=draft_summary_with_langchain(result)
            except Exception as err:
                st.warning(f"LLM unavailable; using template report. Details: {err}")
        st.subheader("Screening report")
        st.text(report)
        if summary:st.write("Supplemental LLM narrative:",summary)
        pdf=build_pdf(result,file.name,report,processed,cnn_overlay,trans_overlay,summary)
        st.download_button("Download PDF screening report",data=pdf,file_name="retinal_screening_report.pdf",mime="application/pdf")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)
st.info("Research-only prototype. It cannot diagnose disease; clinician review is essential.")
