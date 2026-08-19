"""Streamlit demo for the Phase 21 validated retrieval service.

Run from the repository root with::

    uv run streamlit run src/omnisearch/ui/streamlit_app.py --server.headless true

The UI is intentionally a thin in-process adapter.  Streamlit's resource
cache owns one :class:`RetrievalService` per process and reruns reuse it.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any

# The local Apple/PyTorch runtime can crash in image inference when it creates
# an unrestricted native thread pool.  Keep the UI process conservative while
# preserving the Phase 21 service's model and index implementation.
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import streamlit as st

from omnisearch.api.config import ServiceConfig
from omnisearch.api.errors import ServiceError
from omnisearch.api.retrieval import RetrievalService
from omnisearch.ui.adapter import (
    UI_MAX_TOP_K,
    UI_MIN_TOP_K,
    format_results,
    run_image_search,
    run_text_search,
    safe_error_message,
)
from omnisearch.ui.cloud_demo import CompactCloudDemoService, local_artifacts_available

MODE_TEXT_TO_IMAGE = "text-to-image"
MODE_IMAGE_TO_TEXT = "image-to-text"
MODE_LABELS = {
    MODE_TEXT_TO_IMAGE: "Text → Image",
    MODE_IMAGE_TO_TEXT: "Image → Captions",
}
EXAMPLE_QUERIES = (
    "a person riding a bicycle",
    "a dog playing outside",
    "people sitting at a table",
    "a street scene",
)

_ACTIVE_SERVICES: list[Any] = []


def _close_services() -> None:
    """Drop model/index references when the Streamlit process exits."""

    seen: set[int] = set()
    for service in _ACTIVE_SERVICES:
        if id(service) in seen:
            continue
        seen.add(id(service))
        close = getattr(service, "close", None)
        if callable(close):
            close()


atexit.register(_close_services)


@st.cache_resource(show_spinner=False)
def load_service(root: str) -> Any:
    """Load the full service or its artifact-free CPU cloud fallback once."""

    config = ServiceConfig.from_env(Path(root))
    cloud_demo_requested = os.environ.get("OMNISEARCH_CLOUD_DEMO", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if cloud_demo_requested or not local_artifacts_available(config):
        service: Any = CompactCloudDemoService(config)
    else:
        service = RetrievalService(config)
    service.load()
    if not any(existing is service for existing in _ACTIVE_SERVICES):
        _ACTIVE_SERVICES.append(service)
    return service


def _safe_image_path(config: ServiceConfig, filename: str) -> Path | None:
    if not filename:
        return None
    root = config.image_root.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _render_status(config: ServiceConfig, service: Any | None, service_error: BaseException | None) -> None:
    st.sidebar.subheader("Service status")
    if service is None:
        st.sidebar.error("Not ready")
        st.sidebar.caption(safe_error_message(service_error or ServiceError("not ready")))
        status = {
            "model": config.model_id,
            "backend": "FAISS Flat exact inner-product search",
            "device": config.device,
            "ready": False,
            "default_top_k": config.default_top_k,
            "API version": config.api_version,
        }
    else:
        info = service.info()
        st.sidebar.success("Ready")
        status = {
            "model": info.get("model_display_name", info.get("model_id")),
            "backend": info.get("retrieval_backend"),
            "device": info.get("device"),
            "ready": service.ready,
            "default_top_k": info.get("default_top_k"),
            "API version": info.get("api_version"),
        }
        if info.get("deployment_mode"):
            status["deployment mode"] = info["deployment_mode"]
        if info.get("gallery_note"):
            status["gallery note"] = info["gallery_note"]
    st.sidebar.json(status)


def _render_latency(response: dict[str, Any]) -> None:
    latency = response.get("latency_ms", {})
    with st.expander("Latency", expanded=False):
        st.caption("Backend timings come from the reused Phase 21 service; UI service-call timing includes adapter overhead.")
        st.write(
            {
                "preprocessing_ms": round(float(latency.get("preprocessing_ms", 0.0)), 3),
                "query_encoding_ms": round(float(latency.get("query_encoding_ms", 0.0)), 3),
                "search_ms": round(float(latency.get("search_ms", 0.0)), 3),
                "backend_total_ms": round(float(latency.get("total_server_ms", 0.0)), 3),
                "ui_service_call_wall_ms": round(float(latency.get("ui_service_call_wall_ms", 0.0)), 3),
                "ui_adapter_overhead_ms": round(float(latency.get("ui_adapter_overhead_ms", 0.0)), 3),
            }
        )


def _render_image_results(config: ServiceConfig, response: dict[str, Any], service: Any | None = None) -> None:
    rows = format_results(response, MODE_TEXT_TO_IMAGE)
    if not rows:
        st.info("No image results were returned.")
        return
    columns = st.columns(min(3, len(rows)))
    for index, row in enumerate(rows):
        with columns[index % len(columns)]:
            image_path = _safe_image_path(config, row["filename"])
            if image_path is not None:
                st.image(str(image_path), width="stretch")
            else:
                preview_loader = getattr(service, "get_image_preview", None)
                try:
                    preview = preview_loader(row["image_id"]) if callable(preview_loader) else None
                except (OSError, RuntimeError, TypeError, ValueError):
                    preview = None
                if preview is not None:
                    st.image(preview, width="stretch")
                else:
                    st.warning("Image preview unavailable")
            st.markdown(f"**Rank {row['rank']} · score {row['score']:.4f}**")
            st.caption(f"Image ID: {row['image_id']} · {row['filename'] or 'filename unavailable'}")
    _render_latency(response)


def _render_caption_results(response: dict[str, Any]) -> None:
    rows = format_results(response, MODE_IMAGE_TO_TEXT)
    if not rows:
        st.info("No caption results were returned.")
        return
    for row in rows:
        st.markdown(f"**Rank {row['rank']} · score {row['score']:.4f}**")
        st.write(row["text"] or "Caption text unavailable")
        st.caption(f"Caption ID: {row['caption_id']} · Source image ID: {row['image_id']}")
    _render_latency(response)


def _render_about(deployment_mode: str | None = None) -> None:
    with st.expander("About and limitations", expanded=False):
        if deployment_mode == "compact_cloud_demo":
            st.write("This public deployment uses a compact public COCO validation gallery and the public zero-shot CLIP checkpoint; it is not the full validated COCO benchmark.")
        else:
            st.write("Research/demo interface over the validated COCO retrieval artifact; it is not a production service.")
            st.write("The corpus is COCO 2017 validation data with five captions per image. Scores are retrieval similarities, not calibrated confidence.")
        st.write("No authentication, rate limiting, moderation, or private-data audit is provided.")
        st.write("CONTENT-SAFETY FILTERING: NOT IMPLEMENTED")
        st.write("Uploaded images are decoded in memory for the request and are not written to disk by this UI.")


def main() -> None:
    st.set_page_config(page_title="OmniSearch", page_icon="🔎", layout="wide")
    config = ServiceConfig.from_env(Path.cwd())
    st.title("OmniSearch")
    st.caption("Multimodal semantic retrieval demo · Phase 22")
    st.write("Search an image-text retrieval gallery in either direction. A local checkout with the validated artifacts uses the full COCO service; an artifact-free public deployment uses a compact CPU demo gallery.")
    st.caption(
        f"Research/demo only · uploads stay in memory · image uploads are limited to "
        f"{config.max_upload_bytes // (1024 * 1024)} MiB and {config.max_image_pixels:,} pixels"
    )

    with st.sidebar:
        st.header("Search controls")
        mode = st.radio("Search mode", list(MODE_LABELS), format_func=MODE_LABELS.get, key="search_mode")
        top_k = st.slider(
            "Top-k results",
            min_value=UI_MIN_TOP_K,
            max_value=UI_MAX_TOP_K,
            value=min(config.default_top_k, UI_MAX_TOP_K),
            key="top_k",
        )
        if mode == MODE_TEXT_TO_IMAGE:
            st.caption("Examples")
            for example in EXAMPLE_QUERIES:
                if st.button(example, key=f"example_{example}", use_container_width=True):
                    st.session_state["current_query"] = example

    service: Any | None = None
    service_error: BaseException | None = None
    with st.spinner("Loading validated model and indexes…"):
        try:
            service = load_service(str(config.root))
        except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError, ServiceError) as error:
            service_error = error
    active_config = getattr(service, "config", config)
    deployment_mode = None
    if service is not None:
        deployment_mode = service.info().get("deployment_mode")
    _render_status(active_config, service, service_error)

    if service is None:
        st.error("Retrieval service unavailable. Check the local artifacts or the public model download and restart the app.")
        _render_about()
        return

    if mode == MODE_TEXT_TO_IMAGE:
        query = st.text_input("Text query", key="current_query", placeholder="Describe an image…")
        if st.button("Search images", type="primary", use_container_width=True):
            try:
                with st.spinner("Encoding query and searching images…"):
                    response = run_text_search(service, query, top_k)
                st.session_state["latest_results"] = response
                st.session_state["latest_mode"] = MODE_TEXT_TO_IMAGE
            except (OSError, RuntimeError, TypeError, ValueError, ServiceError) as error:
                st.error(safe_error_message(error))
    else:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key="image_upload")
        if uploaded is not None:
            if uploaded.size > config.max_upload_bytes:
                st.error(f"This upload is too large. Maximum size is {config.max_upload_bytes // (1024 * 1024)} MiB.")
            else:
                st.image(uploaded, caption="Uploaded image (kept in memory for this request)", width=240)
        if st.button("Search captions", type="primary", use_container_width=True):
            if uploaded is None:
                st.warning("Upload a JPEG, PNG, or WEBP image first.")
            elif uploaded.size > config.max_upload_bytes:
                st.warning("Choose a smaller image before searching.")
            else:
                try:
                    with st.spinner("Encoding image and searching captions…"):
                        response = run_image_search(service, uploaded.getvalue(), top_k)
                    st.session_state["latest_results"] = response
                    st.session_state["latest_mode"] = MODE_IMAGE_TO_TEXT
                except (OSError, RuntimeError, TypeError, ValueError, ServiceError) as error:
                    st.error(safe_error_message(error))

    latest = st.session_state.get("latest_results")
    latest_mode = st.session_state.get("latest_mode")
    if isinstance(latest, dict) and latest_mode == mode:
        st.subheader("Results")
        if mode == MODE_TEXT_TO_IMAGE:
            _render_image_results(active_config, latest, service)
        else:
            _render_caption_results(latest)
    _render_about(deployment_mode)


if __name__ == "__main__":
    main()
