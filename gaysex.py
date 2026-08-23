from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Tuple, Any

import pymupdf
import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image, ImageDraw


# ============================================================
# Configuration
# ============================================================

APP_TITLE = "KYC & Citizenship Document Validator"
ROI_FILE = Path("saved_rois.json")

DOCUMENT_TYPES = {
    "Citizenship Certificate": {
        "id": "citizenship",
        "fields": {
            "Full Name (नाम थर)": {
                "key": "full_name",
                "default": (0, 0, 0, 0),
            },
            "Citizenship Number (ना. प्र. नं.)": {
                "key": "citizenship_number",
                "default": (0, 0, 0, 0),
            },
            "Date of Birth (जन्म मिति)": {
                "key": "date_of_birth",
                "default": (0, 0, 0, 0),
            },
            "Permanent Address (स्थायी बासस्थान)": {
                "key": "permanent_address",
                "default": (0, 0, 0, 0),
            },
        },
    },
    "Siddhartha Bank KYC Form (Page 1)": {
        "id": "kyc",
        "fields": {
            "Applicant Name (निवेदकको नाम)": {
                "key": "applicant_name",
                "default": (0, 0, 0, 0),
            },
            "Date of Birth (जन्म मिति)": {
                "key": "date_of_birth",
                "default": (0, 0, 0, 0),
            },
            "Citizenship No. (नागरिकता नं.)": {
                "key": "citizenship_number",
                "default": (0, 0, 0, 0),
            },
            "Issue District (जारी भएको जिल्ला)": {
                "key": "issue_district",
                "default": (0, 0, 0, 0),
            },
        },
    },
}


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Session state
# ============================================================

def initialise_session_state() -> None:
    defaults = {
        "document_type": "Citizenship Certificate",
        "uploaded_file_name": None,
        "document_bytes": None,
        "document_image": None,
        "selected_pdf_page": 0,
        "ocr_results": {},
        "manual_values": {},
        "document_legible": False,
        "details_match": False,
        "kyc_approved": False,
        "last_ocr_signature": None,
        "status_message": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_state()


# ============================================================
# ROI persistence
# ============================================================

def load_roi_file() -> Dict[str, Any]:
    """Safely load saved ROI configurations."""
    if not ROI_FILE.exists():
        return {}

    try:
        with ROI_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return {}

        return data

    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def save_roi_file(data: Dict[str, Any]) -> bool:
    """Safely write ROI configuration."""
    try:
        temporary_file = ROI_FILE.with_suffix(".tmp")

        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        temporary_file.replace(ROI_FILE)
        return True

    except (OSError, TypeError, ValueError):
        return False


def template_id(document_type: str) -> str:
    return DOCUMENT_TYPES[document_type]["id"]


def get_saved_rois(document_type: str) -> Dict[str, Tuple[int, int, int, int]]:
    """Return validated saved ROIs for a template."""
    all_rois = load_roi_file()
    doc_id = template_id(document_type)

    saved = all_rois.get(doc_id, {})
    if not isinstance(saved, dict):
        return {}

    result = {}

    for field_name, field_info in DOCUMENT_TYPES[document_type]["fields"].items():
        raw = saved.get(field_info["key"])

        if (
            isinstance(raw, list)
            and len(raw) == 4
            and all(isinstance(value, (int, float)) for value in raw)
        ):
            try:
                result[field_info["key"]] = tuple(
                    max(0, int(value)) for value in raw
                )
            except (ValueError, TypeError):
                pass

    return result


def initialise_template_rois(document_type: str, image: Image.Image | None) -> None:
    """
    Load saved ROIs into session state.

    Saved ROIs are stored in pixels. If there are no saved ROIs,
    zero-sized ROIs are used rather than inventing coordinates for
    a document whose actual scan dimensions are unknown.
    """
    saved = get_saved_rois(document_type)

    for field_name, field_info in DOCUMENT_TYPES[document_type]["fields"].items():
        key = field_info["key"]
        state_key = roi_state_key(document_type, key)

        if state_key in st.session_state:
            continue

        if key in saved:
            st.session_state[state_key] = saved[key]
        else:
            default = field_info["default"]

            if image is not None and default == (0, 0, 0, 0):
                # Intentionally leave empty. The operator can draw/configure
                # the actual ROI for the supplied bank/citizenship template.
                st.session_state[state_key] = (0, 0, 0, 0)
            else:
                st.session_state[state_key] = default


def roi_state_key(document_type: str, field_key: str) -> str:
    return f"roi_{template_id(document_type)}_{field_key}"


def get_roi(document_type: str, field_key: str) -> Tuple[int, int, int, int]:
    value = st.session_state.get(
        roi_state_key(document_type, field_key),
        (0, 0, 0, 0),
    )

    try:
        if len(value) != 4:
            return 0, 0, 0, 0

        return tuple(max(0, int(v)) for v in value)  # type: ignore
    except (TypeError, ValueError):
        return 0, 0, 0, 0


# ============================================================
# Document loading
# ============================================================

def render_pdf_page(pdf_bytes: bytes, page_number: int) -> Image.Image:
    """Render exactly one PDF page into a PIL image."""
    if not pdf_bytes:
        raise ValueError("The uploaded PDF contains no data.")

    document = None

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        if document.page_count == 0:
            raise ValueError("The PDF contains no pages.")

        if page_number < 0 or page_number >= document.page_count:
            raise IndexError("Selected PDF page is outside the document.")

        page = document.load_page(page_number)

        # 2x rendering gives Tesseract considerably more useful pixels.
        matrix = pymupdf.Matrix(2.0, 2.0)
        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
            colorspace=pymupdf.csRGB,
        )

        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples,
        )

        return image

    finally:
        if document is not None:
            document.close()


def load_uploaded_document(
    uploaded_file,
    pdf_page: int = 0,
) -> Image.Image:
    """Load a PDF or image safely."""
    if uploaded_file is None:
        raise ValueError("No document has been uploaded.")

    raw_bytes = uploaded_file.getvalue()

    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return render_pdf_page(raw_bytes, pdf_page)

    if file_name.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()

        return image.convert("RGB")

    raise ValueError(
        "Unsupported file type. Please upload a PDF, PNG, JPG, or JPEG."
    )


# ============================================================
# Image / ROI processing
# ============================================================

def clamp_roi(
    roi: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int] | None:
    """
    Clamp ROI to image boundaries.

    Returns None for a zero-sized or completely invalid ROI.
    """
    try:
        x, y, w, h = [int(v) for v in roi]
    except (TypeError, ValueError):
        return None

    if w <= 0 or h <= 0:
        return None

    x = max(0, x)
    y = max(0, y)

    if x >= image_width or y >= image_height:
        return None

    right = min(image_width, x + w)
    bottom = min(image_height, y + h)

    final_width = right - x
    final_height = bottom - y

    if final_width <= 0 or final_height <= 0:
        return None

    return x, y, final_width, final_height


def crop_roi(
    image: Image.Image,
    roi: Tuple[int, int, int, int],
) -> Image.Image | None:
    """Safely crop an ROI."""
    bounded = clamp_roi(roi, image.width, image.height)

    if bounded is None:
        return None

    x, y, w, h = bounded
    return image.crop((x, y, x + w, y + h))


def preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    """
    Prepare ROI for Devanagari OCR.

    Uses grayscale + upscale + light denoising + adaptive thresholding.
    """
    rgb = np.array(image.convert("RGB"))

    if rgb.size == 0:
        raise ValueError("ROI contains no image data.")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Upscaling helps small Nepali characters.
    gray = cv2.resize(
        gray,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return processed


def get_available_tesseract_languages() -> list[str]:
    try:
        languages = pytesseract.get_languages(config="")
        return [str(language) for language in languages]
    except Exception:
        return []


def choose_ocr_language() -> str:
    """
    Prefer Nepali + English.

    If the Nepali Tesseract language pack is unavailable, fall back
    to English rather than crashing the demo.
    """
    try:
        languages = get_available_tesseract_languages()

        if "nep" in languages and "eng" in languages:
            return "nep+eng"

        if "nep" in languages:
            return "nep"

        if "eng" in languages:
            return "eng"

    except Exception:
        pass

    return "eng"


def perform_ocr(roi_image: Image.Image) -> str:
    """Run OCR with defensive fallback."""
    processed = preprocess_for_ocr(roi_image)

    language = choose_ocr_language()

    config = (
        "--oem 3 "
        "--psm 7 "
        "-c preserve_interword_spaces=1"
    )

    try:
        text = pytesseract.image_to_string(
            processed,
            lang=language,
            config=config,
        )

        return " ".join(text.split()).strip()

    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR is not installed or is not available on PATH."
        )

    except pytesseract.TesseractError as exc:
        raise RuntimeError(f"Tesseract OCR error: {exc}")

    except Exception as exc:
        raise RuntimeError(f"OCR failed: {exc}")


# ============================================================
# Preview
# ============================================================

def create_preview(
    image: Image.Image,
    document_type: str,
) -> Image.Image:
    """Draw every valid ROI on a copy of the document."""
    preview = image.copy().convert("RGB")
    draw = ImageDraw.Draw(preview)

    for index, (field_name, field_info) in enumerate(
        DOCUMENT_TYPES[document_type]["fields"].items()
    ):
        roi = get_roi(document_type, field_info["key"])
        bounded = clamp_roi(roi, preview.width, preview.height)

        if bounded is None:
            continue

        x, y, w, h = bounded

        draw.rectangle(
            [x, y, x + w, y + h],
            outline="red",
            width=max(2, min(preview.width, preview.height) // 500),
        )

        label = str(index + 1)

        # Keep labels visible even on small scans.
        text_y = max(0, y - 22)

        draw.rectangle(
            [x, text_y, x + 28, y],
            fill="red",
        )

        draw.text(
            (x + 7, text_y + 3),
            label,
            fill="white",
        )

    return preview


# ============================================================
# ROI controls
# ============================================================

def render_roi_controls(document_type: str, image: Image.Image) -> None:
    """Render inline x/y/w/h controls."""
    st.subheader("ROI Configuration")

    st.caption(
        f"Image resolution: {image.width} × {image.height} px. "
        "Coordinates are stored in pixels."
    )

    for index, (field_name, field_info) in enumerate(
        DOCUMENT_TYPES[document_type]["fields"].items(),
        start=1,
    ):
        key = field_info["key"]
        state_key = roi_state_key(document_type, key)

        current = get_roi(document_type, key)

        st.markdown(f"**{index}. {field_name}**")

        col_x, col_y, col_w, col_h = st.columns(4)

        with col_x:
            x = st.number_input(
                "X",
                min_value=0,
                max_value=max(0, image.width - 1),
                value=min(current[0], max(0, image.width - 1)),
                step=1,
                key=f"{state_key}_x",
            )

        with col_y:
            y = st.number_input(
                "Y",
                min_value=0,
                max_value=max(0, image.height - 1),
                value=min(current[1], max(0, image.height - 1)),
                step=1,
                key=f"{state_key}_y",
            )

        max_width = max(1, image.width - int(x))
        max_height = max(1, image.height - int(y))

        with col_w:
            w = st.number_input(
                "Width",
                min_value=0,
                max_value=max_width,
                value=min(current[2], max_width),
                step=1,
                key=f"{state_key}_w",
            )

        with col_h:
            h = st.number_input(
                "Height",
                min_value=0,
                max_value=max_height,
                value=min(current[3], max_height),
                step=1,
                key=f"{state_key}_h",
            )

        st.session_state[state_key] = (
            int(x),
            int(y),
            int(w),
            int(h),
        )

        bounded = clamp_roi(
            (int(x), int(y), int(w), int(h)),
            image.width,
            image.height,
        )

        if bounded is None:
            st.warning(
                f"ROI for '{field_name}' is empty or outside the image."
            )


def save_current_rois(document_type: str) -> bool:
    """Save active template ROIs."""
    data = load_roi_file()
    doc_id = template_id(document_type)

    data.setdefault(doc_id, {})

    for field_name, field_info in DOCUMENT_TYPES[document_type]["fields"].items():
        key = field_info["key"]
        data[doc_id][key] = list(get_roi(document_type, key))

    return save_roi_file(data)


# ============================================================
# OCR and results
# ============================================================

def run_document_ocr(
    image: Image.Image,
    document_type: str,
) -> Dict[str, str]:
    """OCR every configured field independently."""
    results: Dict[str, str] = {}

    for field_name, field_info in DOCUMENT_TYPES[document_type]["fields"].items():
        key = field_info["key"]
        roi = get_roi(document_type, key)

        try:
            cropped = crop_roi(image, roi)

            if cropped is None:
                results[key] = ""
                st.warning(
                    f"Skipping '{field_name}': ROI is invalid or empty."
                )
                continue

            results[key] = perform_ocr(cropped)

        except Exception as exc:
            results[key] = ""
            st.error(
                f"OCR failed for '{field_name}': {exc}"
            )

    return results


def build_json_summary(
    document_type: str,
    file_name: str | None,
) -> Dict[str, Any]:
    fields = {}

    for field_name, field_info in DOCUMENT_TYPES[document_type]["fields"].items():
        key = field_info["key"]

        manual_value = st.session_state["manual_values"].get(key, "")
        ocr_value = st.session_state["ocr_results"].get(key, "")

        fields[key] = {
            "label": field_name,
            "ocr_value": ocr_value,
            "verified_value": manual_value,
            "roi": list(get_roi(document_type, key)),
        }

    return {
        "application": APP_TITLE,
        "document_type": document_type,
        "file_name": file_name,
        "single_page_processing": True,
        "verification": {
            "document_legible": bool(
                st.session_state["document_legible"]
            ),
            "details_match": bool(
                st.session_state["details_match"]
            ),
            "kyc_approved": bool(
                st.session_state["kyc_approved"]
            ),
        },
        "fields": fields,
    }


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("🏦 KYC Validator")

    selected_document_type = st.radio(
        "Document Type",
        options=list(DOCUMENT_TYPES.keys()),
        key="document_type",
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload one PDF or image. PDFs are processed one page at a time.",
    )


# ============================================================
# Main application
# ============================================================

st.title(APP_TITLE)

st.caption(
    "Single-page OCR and verification prototype for Nepali Citizenship "
    "Certificates and Siddhartha Bank KYC Page 1."
)


# ------------------------------------------------------------
# Handle uploaded file
# ------------------------------------------------------------

if uploaded_file is not None:
    try:
        file_changed = (
            st.session_state["uploaded_file_name"]
            != uploaded_file.name
        )

        if file_changed:
            st.session_state["document_bytes"] = uploaded_file.getvalue()
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.session_state["document_image"] = None
            st.session_state["ocr_results"] = {}
            st.session_state["manual_values"] = {}
            st.session_state["last_ocr_signature"] = None

        # PDF page selector.
        is_pdf = uploaded_file.name.lower().endswith(".pdf")

        if is_pdf:
            try:
                pdf_document = pymupdf.open(
                    stream=st.session_state["document_bytes"],
                    filetype="pdf",
                )
                page_count = pdf_document.page_count
                pdf_document.close()

            except Exception as exc:
                st.error(f"Unable to inspect PDF: {exc}")
                page_count = 0

            if page_count > 0:
                selected_page = st.sidebar.selectbox(
                    "PDF Page",
                    options=list(range(1, page_count + 1)),
                    index=min(
                        st.session_state["selected_pdf_page"],
                        page_count - 1,
                    ),
                    format_func=lambda page: f"Page {page}",
                )

                st.session_state["selected_pdf_page"] = selected_page - 1

        else:
            page_count = 1
            st.session_state["selected_pdf_page"] = 0

        # Load selected page/image.
        image_signature = (
            uploaded_file.name,
            st.session_state["selected_pdf_page"],
        )

        if (
            st.session_state["document_image"] is None
            or st.session_state.get("loaded_image_signature")
            != image_signature
        ):
            st.session_state["document_image"] = load_uploaded_document(
                uploaded_file,
                st.session_state["selected_pdf_page"],
            )

            st.session_state["loaded_image_signature"] = image_signature

            # Preserve user text when only the PDF page changes if possible.
            st.session_state["ocr_results"] = {}
            st.session_state["last_ocr_signature"] = None

    except Exception as exc:
        st.error(f"Could not load document: {exc}")
        st.stop()


image = st.session_state.get("document_image")


if image is None:
    st.info(
        "Upload a single-page image or PDF to begin. "
        "For multi-page PDFs, select one page at a time."
    )
    st.stop()


# ------------------------------------------------------------
# Load template-specific ROIs
# ------------------------------------------------------------

initialise_template_rois(
    st.session_state["document_type"],
    image,
)


# ------------------------------------------------------------
# Two-column executive layout
# ------------------------------------------------------------

left_column, right_column = st.columns(
    [1.25, 1],
    gap="large",
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_column:
    st.subheader("Document Preview")

    try:
        preview = create_preview(
            image,
            st.session_state["document_type"],
        )

        st.image(
            preview,
            caption=(
                f"{st.session_state['uploaded_file_name']} | "
                f"{image.width} × {image.height}px"
            ),
            use_container_width=True,
        )

        st.caption(
            "Red boxes indicate configured ROIs. "
            "The numbers correspond to the field list below."
        )

    except Exception as exc:
        st.error(f"Unable to render document preview: {exc}")

    render_roi_controls(
        st.session_state["document_type"],
        image,
    )

    roi_col1, roi_col2 = st.columns(2)

    with roi_col1:
        if st.button(
            "💾 Save ROI Config",
            use_container_width=True,
        ):
            try:
                if save_current_rois(
                    st.session_state["document_type"]
                ):
                    st.success(
                        f"ROI configuration saved to {ROI_FILE}."
                    )
                else:
                    st.error("Could not save ROI configuration.")

            except Exception as exc:
                st.error(f"ROI save failed: {exc}")

    with roi_col2:
        if st.button(
            "🔄 Reload Saved ROIs",
            use_container_width=True,
        ):
            try:
                saved = get_saved_rois(
                    st.session_state["document_type"]
                )

                for field_name, field_info in DOCUMENT_TYPES[
                    st.session_state["document_type"]
                ]["fields"].items():
                    key = field_info["key"]

                    if key in saved:
                        st.session_state[
                            roi_state_key(
                                st.session_state["document_type"],
                                key,
                            )
                        ] = saved[key]

                st.success("Saved ROI configuration reloaded.")
                st.rerun()

            except Exception as exc:
                st.error(f"Could not reload ROI configuration: {exc}")


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_column:
    st.subheader("OCR & Verification")

    extract_clicked = st.button(
        "🔎 Extract OCR",
        type="primary",
        use_container_width=True,
    )

    if extract_clicked:
        try:
            with st.spinner("Running OCR on configured fields..."):
                results = run_document_ocr(
                    image,
                    st.session_state["document_type"],
                )

            st.session_state["ocr_results"] = results

            # Initialise editable values from OCR only if the user
            # has not manually modified the corresponding value.
            for key, value in results.items():
                if key not in st.session_state["manual_values"]:
                    st.session_state["manual_values"][key] = value
                elif not st.session_state["manual_values"][key]:
                    st.session_state["manual_values"][key] = value

            st.session_state["last_ocr_signature"] = (
                st.session_state["uploaded_file_name"],
                st.session_state["document_type"],
                tuple(
                    get_roi(
                        st.session_state["document_type"],
                        field_info["key"],
                    )
                    for field_name, field_info in DOCUMENT_TYPES[
                        st.session_state["document_type"]
                    ]["fields"].items()
                ),
            )

            st.success("OCR processing completed.")

        except Exception as exc:
            st.error(f"OCR processing failed: {exc}")


    # --------------------------------------------------------
    # Editable fields
    # --------------------------------------------------------

    st.markdown("### Extracted Details")

    current_doc_type = st.session_state["document_type"]

    for field_name, field_info in DOCUMENT_TYPES[
        current_doc_type
    ]["fields"].items():

        key = field_info["key"]

        ocr_value = st.session_state["ocr_results"].get(
            key,
            "",
        )

        if key not in st.session_state["manual_values"]:
            st.session_state["manual_values"][key] = ocr_value

        st.text_input(
            field_name,
            value=st.session_state["manual_values"].get(key, ""),
            key=f"manual_{template_id(current_doc_type)}_{key}",
            on_change=None,
        )

        # Synchronise widget value back into our canonical state.
        st.session_state["manual_values"][key] = st.session_state[
            f"manual_{template_id(current_doc_type)}_{key}"
        ]

        if ocr_value:
            st.caption(f"OCR: {ocr_value}")
        else:
            st.caption("OCR: No text extracted.")


    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    st.markdown("### Verification")

    st.session_state["document_legible"] = st.checkbox(
        "☑️ Document Legible",
        value=st.session_state["document_legible"],
        key="verification_document_legible",
    )

    st.session_state["details_match"] = st.checkbox(
        "☑️ Details Match",
        value=st.session_state["details_match"],
        key="verification_details_match",
    )

    st.session_state["kyc_approved"] = st.checkbox(
        "☑️ KYC Approved",
        value=st.session_state["kyc_approved"],
        key="verification_kyc_approved",
    )


    # --------------------------------------------------------
    # Verification summary
    # --------------------------------------------------------

    st.markdown("### Status")

    checks = [
        st.session_state["document_legible"],
        st.session_state["details_match"],
        st.session_state["kyc_approved"],
    ]

    if all(checks):
        st.success("Verification state: APPROVED")
    elif any(checks):
        st.warning("Verification state: IN REVIEW")
    else:
        st.info("Verification state: NOT VERIFIED")


    # --------------------------------------------------------
    # JSON export
    # --------------------------------------------------------

    st.markdown("### JSON Summary")

    try:
        summary = build_json_summary(
            current_doc_type,
            st.session_state["uploaded_file_name"],
        )

        json_string = json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )

        st.code(
            json_string,
            language="json",
        )

        st.download_button(
            label="⬇️ Download JSON Summary",
            data=json_string.encode("utf-8"),
            file_name="kyc_validation_result.json",
            mime="application/json",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Unable to create JSON summary: {exc}")


# ============================================================
# Footer / operational notes
# ============================================================

st.divider()

st.caption(
    "Prototype note: OCR output must be reviewed by an authorised operator. "
    "This application does not independently establish authenticity, identity, "
    "or legal validity of a citizenship certificate."
)
