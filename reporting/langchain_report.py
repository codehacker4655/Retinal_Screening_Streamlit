
# ============================================================
# RetinaAI: LangChain + Groq GPT-OSS-20B
# Automated Retinal Screening Report Generator
# ============================================================

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()


def draft_summary_with_langchain(result):
    """
    Generate a cautious retinal screening narrative
    using H3 hybrid model predictions.

    Expected result:
    {
        "predicted_class": "DR",
        "probabilities": {
            "Normal": 0.05,
            "DR": 0.80,
            "ARMD": 0.05,
            "Drusen": 0.05,
            "ODC": 0.05
        }
    }
    """

    # --------------------------------------------------------
    # 1. Read Groq API configuration
    # --------------------------------------------------------

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from the .env file."
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    # --------------------------------------------------------
    # 2. Initialize Groq LLM
    # --------------------------------------------------------

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_tokens=450,
        timeout=60,
        max_retries=1,
        reasoning_effort="low"
    )

    # --------------------------------------------------------
    # 3. Define safe report-generation prompt
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a professional report-writing assistant for a
university retinal screening research prototype.

The screening predictions come from an experimental
five-class Hybrid EfficientNet-B2 + Transformer model.

Your task is to convert the supplied classification
results into a concise screening summary.

STRICT RULES:

1. Use ONLY the supplied classifier outputs.
2. Do NOT independently diagnose any disease.
3. Do NOT invent retinal lesions, anatomical findings,
   disease severity, symptoms, or clinical observations.
4. Do NOT recommend medication or treatment.
5. Describe the percentages as model-assigned scores,
   not medically verified diagnostic probabilities.
6. Do NOT interpret Grad-CAM as confirmed lesion location.
7. Clearly state the model's predicted screening category.
8. Acknowledge uncertainty and model limitations.
9. Recommend interpretation by a qualified
   eye-care professional.
10. Write 3 to 5 clear, professional sentences.
11. Do NOT include internal reasoning, markdown headings,
    or unrelated explanations.
12. Never claim that the patient definitely has a disease.

This is a research screening result, NOT a clinical diagnosis.

Return ONLY the screening narrative.
"""
        ),
        (
            "human",
            """
Research model:
H3 Hybrid EfficientNet-B2 + Transformer

Predicted screening category:
{category}

Model-assigned class scores:
{scores}

CNN Grad-CAM and Transformer attribution visualizations
are generated separately and are not clinical findings.

Generate the retinal screening summary.
"""
        )
    ])

    # --------------------------------------------------------
    # 4. Validate the supplied prediction data
    # --------------------------------------------------------

    if "predicted_class" not in result:
        raise ValueError(
            "Missing 'predicted_class' in model result."
        )

    if "probabilities" not in result:
        raise ValueError(
            "Missing 'probabilities' in model result."
        )

    category = str(result["predicted_class"])

    class_names = [
        "Normal",
        "DR",
        "ARMD",
        "Drusen",
        "ODC"
    ]

    if category not in class_names:
        raise ValueError(
            f"Unexpected predicted class: {category}"
        )

    probabilities = result["probabilities"]

    if not isinstance(probabilities, dict):
        raise ValueError(
            "Probabilities must be a dictionary."
        )

    if set(probabilities) != set(class_names):
        raise ValueError(
            "Expected scores for all five retinal classes."
        )

    scores = "\n".join(
        f"{name}: {float(probabilities[name]):.2%}"
        for name in class_names
    )

    # --------------------------------------------------------
    # 5. Run LangChain report generation
    # --------------------------------------------------------

    chain = prompt | llm

    response = chain.invoke({
        "category": category,
        "scores": scores
    })

    # --------------------------------------------------------
    # 6. Extract and validate generated summary
    # --------------------------------------------------------

    content = response.content

    if isinstance(content, str):
        summary = content.strip()
    else:
        summary = " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
        ).strip()

    if not summary:
        raise RuntimeError(
            "Groq returned an empty screening summary."
        )

    # Fail safely if reasoning markup appears
    if "<think>" in summary.lower():
        raise RuntimeError(
            "Unexpected reasoning markup in LLM output."
        )

    return summary
