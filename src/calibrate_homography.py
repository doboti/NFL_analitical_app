"""
Interaktív pálya-homográfia kalibráló eszköz.

Használat:
    streamlit run src/calibrate_homography.py

Kattints a referencia-képen (data/reference/calibration_frame.png) legalább
4, jól azonosítható pálya-pontra (pl. oldalvonal és yard-vonal metszéspontja),
majd add meg mindegyikhez a valós pálya-koordinátát (yardline 0-120,
oldaltávolság 0-53.33). Minimum 4 pont kell a homográfiához, de minél több
és minél inkább szétszórt (nem egy vonalban lévő) pont, annál pontosabb
az eredmény.

A "Homográfia mentése" gombra kattintva a mátrix a models/homography.json
fájlba kerül, amit a src/detect_players.py automatikusan felhasznál.
"""
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from homography import FieldHomography, HOMOGRAPHY_PATH

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_IMAGE = ROOT / "data" / "reference" / "calibration_frame.png"

st.set_page_config(page_title="Pálya-homográfia kalibráció", layout="wide")
st.title("🏈 Pálya-homográfia kalibráció")

if not REFERENCE_IMAGE.exists():
    st.error(f"Nincs referencia-kép: {REFERENCE_IMAGE}")
    st.stop()

if "calib_points" not in st.session_state:
    st.session_state.calib_points = []

img = Image.open(REFERENCE_IMAGE)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Kattints egy jól azonosítható pálya-pontra")
    coords = streamlit_image_coordinates(img, key="calib_img")
    if coords is not None:
        candidate = (coords["x"], coords["y"])
        if not st.session_state.calib_points or st.session_state.calib_points[-1][0] != candidate:
            st.session_state.calib_points.append([candidate[0], candidate[1], None, None])

with col2:
    st.subheader("Felvett pontok")
    st.caption("Add meg minden ponthoz a valós pálya-koordinátát: "
               "yardline (0-120, saját gólvonaltól) és oldaltávolság (0-53.33 yard).")

    to_remove = None
    for i, point in enumerate(st.session_state.calib_points):
        px, py, yard_x, yard_y = point
        st.markdown(f"**Pont {i+1}** — pixel: ({px:.0f}, {py:.0f})")
        c1, c2, c3 = st.columns([1, 1, 0.4])
        yard_x = c1.number_input("Yardline (0-120)", 0.0, 120.0, float(yard_x or 50.0), key=f"yx_{i}")
        yard_y = c2.number_input("Oldaltáv. (0-53.33)", 0.0, 53.33, float(yard_y or 26.65), key=f"yy_{i}")
        if c3.button("Törlés", key=f"del_{i}"):
            to_remove = i
        st.session_state.calib_points[i] = [px, py, yard_x, yard_y]

    if to_remove is not None:
        st.session_state.calib_points.pop(to_remove)
        st.rerun()

    if st.button("Összes pont törlése"):
        st.session_state.calib_points = []
        st.rerun()

    st.divider()
    n_points = len(st.session_state.calib_points)
    st.metric("Felvett pontok száma", n_points)

    if n_points >= 4:
        if st.button("Homográfia számítása és mentése", type="primary"):
            pixel_points = [(p[0], p[1]) for p in st.session_state.calib_points]
            field_points = [(p[2], p[3]) for p in st.session_state.calib_points]
            try:
                homography = FieldHomography.from_points(pixel_points, field_points)
                homography.save()
                st.success(f"Homográfia elmentve: {HOMOGRAPHY_PATH}")
            except Exception as exc:
                st.error(f"Hiba a homográfia számításakor: {exc}")
    else:
        st.info("Legalább 4 pont szükséges a homográfia kiszámításához.")

    if HOMOGRAPHY_PATH.exists():
        st.caption(f"✅ Már van mentett homográfia: {HOMOGRAPHY_PATH}")
