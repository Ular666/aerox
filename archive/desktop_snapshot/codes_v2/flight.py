from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st

PATROL = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
SURVEY = [(0, 0), (0, 30), (10, 30), (10, 0), (20, 0), (20, 30), (0, 0)]
CHECKS = ("Propellers and frame checked", "Battery secured", "Route and operating area reviewed",
          "Simulated GPS, link and controller checks reviewed")


def render():
    st.header("FLIGHT")
    st.warning("SIMULATION MODE • Virtual waypoints and telemetry. Actual telemetry: unavailable.")
    mission = st.selectbox("Mission template", ["Scheduled patrol", "Agricultural survey"])
    inspection = st.session_state.get("field-inspection")
    if mission == "Agricultural survey" and inspection:
        st.subheader("Field inspection list")
        st.text(f"{inspection['source']} | {inspection['grid']} x {inspection['grid']} image grid")
        st.caption("Regions refer to the source image. The virtual route below is illustrative.")
        checklist = pd.DataFrame(inspection["cells"]).assign(source_image=inspection["source"], grid_size=inspection["grid"])
        st.dataframe(checklist, hide_index=True)
        st.download_button("Download inspection list", checklist.to_json(orient="records", indent=2),
                           "field_inspection.json", "application/json", on_click="ignore")
    points = PATROL if mission == "Scheduled patrol" else SURVEY
    cols = st.columns(3)
    with cols[0]:
        start = st.time_input("Planned local start")
    with cols[1]:
        interval = st.number_input("Minutes between planned starts", 5, 180, 30)
    with cols[2]:
        scale = st.number_input("Route scale", 1, 4, 1)
    clock = datetime.combine(datetime(2026, 11, 20).date(), start)
    st.caption("Simulated queue • " + " → ".join((clock + timedelta(minutes=i * interval)).strftime("%H:%M")
                                                  for i in range(3)))
    checked = [st.checkbox(item, key=f"{mission}-{item}") for item in CHECKS]
    if not all(checked):
        st.info("Complete the checklist to step through this virtual route.")
    step = st.slider("Completed virtual waypoints", 0, len(points), key=f"{mission}-{scale}-step",
                     disabled=not all(checked))
    step = step if all(checked) else 0
    st.progress(step / len(points), text=f"{step} of {len(points)} waypoints completed")
    table = pd.DataFrame([dict(East=x * scale, North=y * scale, Order=i + 1,
                               Status="Completed" if i < step else "Pending")
                          for i, (x, y) in enumerate(points)])
    base = alt.Chart(table).encode(x=alt.X("East:Q", title="Local east (m)"),
                                   y=alt.Y("North:Q", title="Local north (m)"))
    st.altair_chart(base.mark_line().encode(order="Order:Q")
                   + base.mark_circle(size=125).encode(color="Status:N", tooltip=["Order", "East", "North"]),
                   width="stretch")
    st.dataframe(table, hide_index=True)
    voltage = 16.8 - 0.18 * step
    for col, (label, value) in zip(st.columns(5), [
        ("Simulated battery", f"{voltage:.2f} V"), ("Battery status", "Nominal (virtual)"),
        ("GPS", "Virtual fix"), ("Link", "Virtual link"), ("Controller", "Virtual response")
    ]):
        col.metric(label, value)
    st.caption("Intended controller family: F405. Board model, firmware and connection remain To Be Verified.")

