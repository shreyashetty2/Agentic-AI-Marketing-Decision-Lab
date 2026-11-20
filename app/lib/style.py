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


_EXPLAINERS = {
    "overview": (
        "**Three questions, one tool each.** Will a household use the coupon (Redemption Likelihood)? "
        "What is it worth (Spend Forecast)? Did the coupon make a difference, or would they have bought "
        "anyway (Campaign Uplift)?\n\n"
        "**The green, amber and red badge is the automatic check's verdict.** *Approve* means the tool "
        "passed. *Human needed* means the checks could not settle it, so a person should judge. A result "
        "whose range includes zero could be luck.\n\n"
        "**Where the numbers come from.** Everything here is read from the last run of the agents. This "
        "page never calculates anything itself."),
    1: (
        "**What this answers:** will a mailed household use the coupon?\n\n"
        "**How to read it:** every household gets a chance of redeeming, from 0% to 100%. Pick a campaign "
        "and how much of its list to mail. The page shows how many households that is, how many "
        "redemptions to expect (with a likely range), and who is on the list.\n\n"
        "**Did it work?** These campaigns are finished, so the page also shows how many of the real "
        "redeemers that list would have reached, next to the simple rule of mailing whoever redeemed "
        "before, and mailing at random.\n\n"
        "**Who the list leaves out:** compares each group's share of the list with its share of everyone "
        "mailed. Age Group runs 1 to 6 in order, but the real ages were removed from this anonymised "
        "data, so we cannot say which end is older.\n\n"
        "**What drives it:** past coupon use and shopping habits, not demographics."),
    2: (
        "**What this answers:** if a household is mailed, about how many dollars will it spend on the "
        "campaign's products?\n\n"
        "**How to read it:** each household gets a dollar forecast. These are averages, not guarantees. "
        "The simple rule to beat is a household's usual spend on those products.\n\n"
        "**Be careful:** there were only 254 redemptions to learn from, so treat the forecast as a rough "
        "guide. Segments with few households can swing a lot."),
    3: (
        "**What this answers:** who does mailing actually change? Uplift is the number of extra buyers a "
        "household adds if mailed, compared with not mailing it.\n\n"
        "**How it works:** mailed households are compared with households that were not mailed. The "
        "question is whether the mailing caused the buying, not just whether the two happened together.\n\n"
        "**Sure things:** households that would likely have bought anyway. Mailing them is a likely "
        "wasted send.\n\n"
        "**Be careful:** mailed households were not picked at random, so uplift scores are approximate "
        "and need a person's judgment."),
}


def explainer(key) -> None:
    """A collapsed plain-language guide for someone with no background in this project."""
    with st.expander("New here? What am I looking at?"):
        st.markdown(_EXPLAINERS[key])
