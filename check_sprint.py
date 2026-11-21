from pathlib import Path
import io
import json
import tempfile
from zipfile import ZipFile

from streamlit.testing.v1 import AppTest
from PIL import Image

import agri
import guard

ROOT = Path(__file__).resolve().parent


def check_ui():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    assert not app.exception
    assert app.header[0].value == "GUARD"
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert [m.value for m in app.metric] == ["12", "51"]
    assert len(app.selectbox[1].options) == 12
    app.selectbox[1].select_index(8).run()
    assert not app.exception and app.get("download_button")
    app.sidebar.radio[0].set_value("AGRI").run()
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.metric[0].value == "53.12%"
    assert agri.YIELD_STATUS in [x.value for x in app.info]
    assert app.get("download_button")
    app.sidebar.radio[0].set_value("FLIGHT").run()
    assert not app.exception
    assert app.slider[0].disabled and len(app.dataframe[0].value) == 5
    for check in app.checkbox:
        check.check()
    app.run()
    assert not app.slider[0].disabled
    app.slider[0].set_value(5).run()
    assert not app.exception
    assert set(app.dataframe[0].value["Status"]) == {"Completed"}
    app.checkbox[0].uncheck().run()
    assert set(app.dataframe[0].value["Status"]) == {"Pending"}
    app.selectbox[0].set_value("Agricultural survey").run()
    assert not app.exception and len(app.dataframe[0].value) == 7
    print("PASS: Streamlit GUARD run, evidence selection, report control, AGRI run and report control, FLIGHT checklist gating and both routes")


def check_edge_cases():
    for stem, expect in (("guard_auto_report", 14), ("guard_cpu_report", 14), ("agri_report", 5)):
        path = ROOT / "demo" / "outputs" / f"{stem}.zip"
        with ZipFile(path) as archive:
            assert archive.testzip() is None
            assert len(archive.namelist()) == expect + (1 if stem.startswith("guard") else 0)
    for rgb, coverage in (((100, 100, 100), 0), ((47, 130, 33), 100)):
        buf = io.BytesIO()
        Image.new("RGB", (35, 33), rgb).save(buf, "PNG")
        result = agri.analyze(buf.getvalue(), 4)
        assert result["coverage"] == coverage
        assert sum(cell["total_pixels"] for cell in result["regions"]) == 35 * 33
    with tempfile.TemporaryDirectory() as directory:
        bad_video = Path(directory) / "bad.mp4"
        bad_video.write_bytes(b"invalid mp4")
        try:
            guard.process_video(bad_video, ROOT / "models/rescue.pt", 0.25, 5, "cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("Expected an unreadable input error")
    print("PASS: all report ZIPs readable, whole-grid accounting, 0/100% ExG and invalid MP4 behavior")


if __name__ == "__main__":
    check_ui()
    check_edge_cases()

