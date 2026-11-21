import io,csv,tempfile
from pathlib import Path
from zipfile import ZipFile
from streamlit.testing.v1 import AppTest
import guard
root=Path(__file__).resolve().parent
def run():
    app=AppTest.from_file(str(root/"app.py"), default_timeout=90).run()
    app.checkbox[0].check().run()
    app.button[0].click().run()
    app.button[1].click().run()
    frame=app.selectbox[1].value
    app.radio(key=next(w.key for w in app.radio if w.label=="Review status")).set_value("Needs second review").run()
    app.text_input[0].set_value("Check overlap").run()
    app.selectbox[1].select(next(i for i in app.session_state["guard-result"][1]["evidence"] if i!=frame)).run()
    app.selectbox[1].select(frame).run()
    assert app.text_input[0].value=="Check overlap"
    result=app.session_state["guard-result"][1]
    with ZipFile(io.BytesIO(guard.report(result,"sample.mp4",result["reviews"]))) as z:
        rows=list(csv.DictReader(io.StringIO(z.read("human_review.csv").decode())))
        row=next(r for r in rows if int(r["frame"])==frame)
        assert row["review_status"]=="Needs second review" and row["operator_note"]=="Check overlap" and row["evidence_file"] in z.namelist()
    app.checkbox[0].uncheck().run()
    assert "guard-result" not in app.session_state
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert app.text_input[0].value=="" and not app.exception
    assert guard.review_candidates([dict(frame=i,seconds=t,people=n) for i,t,n in [(0,0,20),(9,1.99,19),(10,2,18),(11,5,17)]])==[0,10,11]
    print("PASS: queue navigation, review persistence, CSV and image linkage, fresh-run isolation, two-second boundary")
if __name__=="__main__":
    run()
