"""ReportLab PDF export with model-produced figures."""
import io
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image as PDFImage,Table,TableStyle,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader

def build_pdf(result, image_id, report_text, processed, cnn, transformer, llm_summary=None):
    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=(595,842),rightMargin=40,leftMargin=40,topMargin=40,bottomMargin=40)
    styles=getSampleStyleSheet(); story=[]
    story.append(Paragraph("Retinal screening — Research prototype",styles['Title']))
    story.append(Spacer(1,12))
    for line in report_text.splitlines():
        if not line.strip():story.append(Spacer(1,7));continue
        from xml.sax.saxutils import escape
        story.append(Paragraph(escape(line),styles['BodyText']))
        story.append(Spacer(1,3))
    if llm_summary:
        story.append(Spacer(1,10));story.append(Paragraph("LLM-generated supplemental summary",styles['Heading2']))
        from xml.sax.saxutils import escape
        story.append(Paragraph(escape(llm_summary),styles['BodyText']))
    story.append(Spacer(1,12));story.append(Paragraph("Model attribution visualizations (not lesion maps)",styles['Heading2']))
    imgs=[]
    for im in (processed,cnn,transformer):
        b=io.BytesIO(); im.save(b,format='PNG');b.seek(0)
        imgs.append(PDFImage(b,width=155,height=155))
    t=Table([imgs],colWidths=[165,165,165]);story.append(t)
    story.append(Paragraph("Left: P2 Retinex   |   Middle: CNN Grad-CAM   |   Right: Transformer token attribution",styles['BodyText']))
    doc.build(story)
    return buffer.getvalue()
