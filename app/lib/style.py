"""Shared look for every page: the Decision Lab's own skin, not default Streamlit chrome.
Colors are the validated palette from the project's dataviz skill (references/palette.md) --
not eyeballed, the same hexes used in .streamlit/config.toml and the chart helpers. Accent is the
palette's own categorical slot 6 (green), in the spirit of the project's sponsor -- not a claim to
match TD's exact brand hex, and no TD logo or wordmark anywhere in this app."""
import streamlit as st

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
ACCENT = "#008300"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
BORDER = "rgba(11,11,11,0.10)"

# Status palette (fixed, never themed): good / warning / critical.
STATUS = {
    "approve": {"color": "#0ca30c", "icon": "✓", "label": "Approved"},
    "human_needed": {"color": "#fab219", "icon": "⚠", "label": "Needs a person"},
    "fail": {"color": "#d03b3b", "icon": "✗", "label": "Failing"},
}


def inject() -> None:
    st.markdown(f"""
    <style>
    [data-testid="stToolbar"], footer, #MainMenu {{ visibility: hidden; height: 0; }}
    .block-container {{ padding-top: 2.5rem; max-width: 1100px; }}
    html, body, [class*="css"] {{
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }}
    h1 {{
        font-weight: 700; letter-spacing: -0.02em; color: {INK};
        border-bottom: 3px solid {ACCENT}; padding-bottom: 0.4rem; margin-bottom: 1.2rem;
    }}
    h2, h3 {{ font-weight: 650; letter-spacing: -0.01em; color: {INK}; }}
    [data-testid="stMetric"] {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 0.9rem 1rem; box-shadow: 0 1px 2px rgba(11,11,11,0.04);
    }}
    [data-testid="stMetricLabel"] {{ color: {INK_SECONDARY}; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 12px; border-color: {BORDER} !important;
        box-shadow: 0 1px 3px rgba(11,11,11,0.05);
    }}
    [data-testid="stCaptionContainer"] {{ color: {INK_SECONDARY}; }}
    [data-testid="stSidebarNav"] li div a {{ font-weight: 500; }}
    </style>
    """, unsafe_allow_html=True)


def status_badge(verdict: str) -> str:
    """An HTML chip: icon + label + color, never color alone (status colors are sub-3:1 contrast
    on light surfaces by the palette's own design -- the icon and text carry the meaning)."""
    s = STATUS.get(verdict, {"color": INK_MUTED, "icon": "?", "label": verdict.replace("_", " ")})
    return (f'<span style="display:inline-flex;align-items:center;gap:0.4rem;'
            f'background:{s["color"]}1a;color:{INK};border:1px solid {s["color"]}55;'
            f'border-radius:999px;padding:0.3rem 0.9rem;font-weight:600;font-size:0.95rem;">'
            f'<span style="color:{s["color"]};font-weight:800;">{s["icon"]}</span> {s["label"]}</span>')
