from __future__ import annotations

import streamlit as st


def render_kpi(title: str, value: str, delta: str, color: str = "#334155") -> None:
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta" style="color:{color};">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(title: str, value: str, color: str = "#0f172a") -> None:
    st.markdown(
        f"<p class='metric-title'>{title}</p><h4 class='metric-value' style='color:{color};'>{value}</h4>",
        unsafe_allow_html=True,
    )


def render_section_start(title: str, color: str, subtitle: str = "") -> None:
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin:0; color:{color};'>{title}</h3>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color:#475569; margin-bottom:20px;'>{subtitle}</p>", unsafe_allow_html=True)


def render_section_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
