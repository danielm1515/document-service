"""Renders the six demo documents as PDFs dated ISSUE_DATE, with the host's Chrome (headless).

The owner's originals are dated 2024 and so are expired under every validity rule; these copies
carry the same content with a current date. Usage (Windows host):
    python demo/make_demo_pdfs.py            # issue date 15.09.2026 -> demo/2026 and tests/fixtures/2026
    python demo/make_demo_pdfs.py 01.12.2026 demo/december
"""
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
ROOT = Path(__file__).resolve().parent.parent

STYLE = """body{font-family:Arial,sans-serif;margin:48px;color:#1b2430}h1{font-size:22px;margin:0}
.org{color:#555;margin-bottom:18px}h2{font-size:19px;margin:18px 0 8px}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:6px 10px;text-align:right}.meta td{border:none;padding:2px 0}"""

HEADER = """<h1>המרכז הרפואי רמון</h1><div class="org">מחלקת מסמכים רפואיים</div>
<table class="meta"><tr><td>תאריך הפקה: {date}</td></tr><tr><td>מידע המטופל: מסווג</td></tr>
<tr><td>מקום ביצוע: המרכז הרפואי רמון</td></tr></table>"""

DOCUMENTS = {
    "cbc": ("01_ספירת_דם_מלאה", "מסמך רפואי", """<h2>תוצאות מעבדה - ספירת דם מלאה</h2>
<table><tr><th>בדיקה</th><th>תוצאה</th><th>טווח ייחוס</th><th>יחידות</th></tr>
<tr><td>תאי דם לבנים</td><td>6.8</td><td>4.0 - 10.0</td><td>אלפים למיקרוליטר</td></tr>
<tr><td>המוגלובין</td><td>14.2</td><td>13.5 - 17.5</td><td>גרם לדציליטר</td></tr>
<tr><td>טסיות דם</td><td>245</td><td>150 - 400</td><td>אלפים למיקרוליטר</td></tr></table>"""),
    "coagulation": ("02_בדיקות_קרישה", "מסמך רפואי", """<h2>תוצאות מעבדה - בדיקות קרישה</h2>
<table><tr><th>בדיקה</th><th>תוצאה</th><th>טווח ייחוס</th><th>יחידות</th></tr>
<tr><td>זמן פרותרומבין</td><td>12.4</td><td>11.0 - 14.0</td><td>שניות</td></tr>
<tr><td>יחס מנורמל בינלאומי</td><td>1.02</td><td>0.90 - 1.20</td><td>יחס</td></tr></table>"""),
    "ecg": ("03_תרשים_לב", "מסמך רפואי", """<h2>תרשים פעילות חשמלית של הלב</h2>
<table><tr><td>סוג הבדיקה</td><td>תרשים פעילות חשמלית של הלב במנוחה</td></tr>
<tr><td>קצב לב</td><td>72 פעימות בדקה</td></tr><tr><td>קצב</td><td>סדיר</td></tr></table>"""),
    "urinalysis": ("04_בדיקת_שתן_לא_רלוונטית", "מסמך רפואי", """<h2>תוצאות מעבדה - בדיקת שתן כללית</h2>
<table><tr><th>מדד</th><th>תוצאה</th><th>טווח ייחוס</th></tr>
<tr><td>משקל סגולי</td><td>1.018</td><td>1.005 - 1.030</td></tr>
<tr><td>חלבון</td><td>שלילי</td><td>שלילי</td></tr></table>"""),
    "preop_summary": ("05_סיכום_טרום_ניתוח", "מסמך סיכום רפואי", """<h2>סיכום הערכה רפואית טרום-ניתוחית</h2>
<table><tr><td>מטרת הביקור</td><td>הערכת מוכנות לקראת הליך ניתוחי מתוכנן</td></tr>
<tr><td>מסמכים שנבדקו</td><td>ספירת דם מלאה, בדיקות קרישה, תרשים פעילות חשמלית של הלב</td></tr></table>"""),
    "electricity_bill": ("06_מסמך_לא_רפואי_חשבון_חשמל", "מסמך שאינו רפואי", """<h2>חשבון חשמל</h2>
<table><tr><td>חברת החשמל האזורית</td><td>חשבון צריכת חשמל תקופתי</td></tr>
<tr><td>סוג המסמך</td><td>חשבון שירות - לא מסמך רפואי</td></tr>
<tr><td>סכום לתשלום</td><td>384.70 ₪</td></tr></table>"""),
}


def render(issue_date: str, out_dirs: list[Path]) -> None:
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, (hebrew_name, kind, body) in DOCUMENTS.items():
            header = HEADER.format(date=issue_date) if name != "electricity_bill" else \
                f"<h1>חשבון חשמל</h1><table class='meta'><tr><td>תאריך הפקה: {issue_date}</td></tr></table>"
            html = (f"<!doctype html><html lang='he' dir='rtl'><head><meta charset='utf-8'><style>{STYLE}</style>"
                    f"</head><body>{header}<p>{kind}</p>{body}<p>מסמך ממוחשב - מסמך דוגמה</p></body></html>")
            source = Path(tmp) / f"{name}.html"
            source.write_text(html, encoding="utf-8")
            target = Path(tmp) / f"{name}.pdf"
            subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                            f"--print-to-pdf={target}", source.as_uri()], check=True, capture_output=True)
            for d in out_dirs:
                file_name = f"{hebrew_name}.pdf" if d.name != "2026" or d.parent.name != "fixtures" else f"{name}.pdf"
                (d / file_name).write_bytes(target.read_bytes())


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else "15.09.2026"
    outs = [Path(sys.argv[2])] if len(sys.argv) > 2 else [ROOT / "demo" / "2026", ROOT / "tests" / "fixtures" / "2026"]
    render(date_arg, outs)
    print(f"rendered {len(DOCUMENTS)} documents dated {date_arg} into", ", ".join(str(o) for o in outs))
