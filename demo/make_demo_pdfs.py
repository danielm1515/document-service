"""Renders the six demo documents as PDFs dated ISSUE_DATE, with the host's Chrome (headless).

The owner's originals are dated 2024 and so are expired under every validity rule; these copies
reproduce the originals' design (teal top bar, organisation header with logo, title card, info
card, results table, note card, footer) with the same content but a current date. Usage
(Windows host):
    python demo/make_demo_pdfs.py            # issue date 15.09.2026 -> demo/2026 and tests/fixtures/2026
    python demo/make_demo_pdfs.py 01.12.2026 demo/december
"""
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Shared look: an inline pulse-icon SVG plus the CSS for every card/table/pill
# the six documents are built from. Kept as plain (non-format) strings because
# the CSS is full of literal braces.
# ---------------------------------------------------------------------------

PULSE_SVG = (
    '<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">'
    '<polyline points="3,21 12,21 16,9 21,31 25,13 29,21 37,21" fill="none" '
    'stroke="#ffffff" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>'
    "</svg>"
)

STYLE = """
:root{
  --teal-dark:#0e5b66; --teal:#1f7a8c; --navy:#1c3240; --gray:#66727e; --gray-light:#8a94a0;
  --line:#e3e8ed; --card-border:#e0e6ec; --title-bg:#e7f2fa; --title-border:#d3e6f4;
  --note-bg:#e2f5ef; --note-border:#bfe4d8; --note-fg:#1c3240;
  --warn-bg:#fdf1d6; --warn-border:#f0d99a; --warn-fg:#5c4413;
  --pill-med-bg:#d9f0e1; --pill-med-fg:#1f7a4d;
  --pill-non-bg:#fbdcdc; --pill-non-fg:#b23a3a;
  --table-head-bg:#eef2f6; --amount-bg:#e7f2fa;
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{font-family:Arial,"Noto Sans Hebrew",sans-serif;color:var(--navy);background:#ffffff;
  -webkit-print-color-adjust:exact;print-color-adjust:exact;font-size:14px;}
.topbar{height:14px;width:100%;background:var(--teal-dark);}
.page{padding:34px 56px 46px;}

.header{display:flex;justify-content:space-between;align-items:flex-start;}
.org-block{display:flex;align-items:flex-start;gap:14px;}
.org-text{text-align:right;}
.org-name{font-size:21px;font-weight:700;line-height:1.25;color:var(--navy);}
.org-dept{font-size:13px;color:var(--gray);margin-top:4px;}
.logo{width:60px;height:60px;border-radius:16px;background:var(--teal);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.logo svg{width:32px;height:32px;}
.issue-date{font-size:13px;color:var(--gray);white-space:nowrap;padding-top:6px;}

.divider{border:none;border-top:1px solid var(--line);margin:22px 0;}

.title-card{background:var(--title-bg);border:1px solid var(--title-border);border-radius:12px;
  padding:22px 26px;margin-bottom:22px;}
.title-card .title{font-size:25px;font-weight:700;color:var(--teal-dark);}
.title-card .subtitle{font-size:13.5px;color:var(--gray);margin-top:6px;}

.info-row{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:26px;flex-wrap:wrap;}
.info-plain{text-align:right;}
.info-plain .label{font-size:12.5px;color:var(--gray);}
.info-plain .value{font-size:14.5px;font-weight:700;margin-top:3px;}
.info-card{border:1px solid var(--card-border);border-radius:10px;padding:12px 20px;
  text-align:center;background:#ffffff;min-width:220px;flex:1 1 auto;}
.info-card .label{font-size:12.5px;color:var(--gray);}
.info-card .value{font-size:14.5px;font-weight:700;margin-top:4px;}
.pill{border-radius:999px;padding:9px 18px;font-size:13.5px;font-weight:700;white-space:nowrap;}
.pill.med{background:var(--pill-med-bg);color:var(--pill-med-fg);}
.pill.non{background:var(--pill-non-bg);color:var(--pill-non-fg);}

.section-heading{font-size:17px;font-weight:700;color:var(--navy);margin:0 0 12px;}

.field-row{display:flex;gap:16px;margin-bottom:16px;}
.field-card{border:1px solid var(--card-border);border-radius:10px;padding:14px 20px;background:#ffffff;
  flex:1;text-align:center;}
.field-card .label{font-size:12.5px;color:var(--gray);}
.field-card .value{font-size:14.5px;font-weight:700;margin-top:5px;}

table{border-collapse:collapse;width:100%;margin-bottom:22px;}
thead th{background:var(--table-head-bg);font-size:13px;font-weight:700;color:var(--navy);
  padding:11px 16px;text-align:center;border-bottom:1px solid var(--line);}
thead th:first-child{text-align:right;}
tbody td{padding:11px 16px;font-size:13.5px;border-bottom:1px solid var(--line);text-align:center;}
tbody td:first-child{text-align:right;font-weight:600;}
tbody tr:last-child td{border-bottom:none;}

.kv-table{border-collapse:collapse;width:100%;margin-bottom:22px;}
.kv-table td{padding:12px 4px;font-size:14px;border-bottom:1px solid var(--line);}
.kv-table td.k{text-align:right;color:var(--navy);}
.kv-table td.v{text-align:left;font-weight:700;}
.kv-table tr:last-child td{border-bottom:none;}

.note{border:1px solid var(--note-border);background:var(--note-bg);color:var(--note-fg);
  border-radius:10px;padding:16px 20px;font-size:13.5px;text-align:right;margin-bottom:22px;line-height:1.6;}
.note.warn{border-color:var(--warn-border);background:var(--warn-bg);color:var(--warn-fg);}

.chart-box{border:1px solid var(--card-border);border-radius:10px;background:#f6fbfd;
  padding:14px;margin-bottom:22px;}

.amount-card{background:var(--amount-bg);border:1px solid var(--title-border);border-radius:12px;
  padding:22px;text-align:center;margin-bottom:22px;}
.amount-card .label{font-size:13px;color:var(--gray);}
.amount-card .value{font-size:34px;font-weight:700;color:var(--teal-dark);margin:8px 0;}
.amount-card .due{font-size:12.5px;color:var(--gray);}

.company-row{display:flex;justify-content:space-between;align-items:flex-start;margin:20px 0 22px;}
.company-name{text-align:right;}
.company-name .name{font-size:20px;font-weight:700;color:var(--navy);}
.company-name .sub{font-size:13px;color:var(--gray);margin-top:5px;}
.company-meta{text-align:left;font-size:12.5px;color:var(--gray);line-height:1.7;}

.sign-row{display:flex;justify-content:space-between;gap:40px;margin:30px 0 6px;}
.sign-box{flex:1;text-align:center;}
.sign-line{border-top:1px solid var(--gray-light);margin-bottom:8px;}
.sign-box .label{font-size:12.5px;color:var(--gray);}

.footer{display:flex;justify-content:space-between;border-top:1px solid var(--line);
  padding-top:12px;margin-top:6px;font-size:11.5px;color:var(--gray-light);}
"""


def _hospital_header(issue_date: str) -> str:
    return f"""<div class="topbar"></div><div class="page"><header class="header">
<div class="org-block"><div class="org-text"><div class="org-name">המרכז הרפואי<br>רמון</div>
<div class="org-dept">מחלקת מסמכים רפואיים</div></div><div class="logo">{PULSE_SVG}</div></div>
<div class="issue-date">תאריך הפקה: {issue_date}</div></header><hr class="divider">"""


def _title_card(title: str, subtitle: str) -> str:
    return f'<div class="title-card"><div class="title">{title}</div><div class="subtitle">{subtitle}</div></div>'


def _info_row(pill_text: str, pill_class: str, location: str = "המרכז הרפואי רמון") -> str:
    return f"""<div class="info-row">
<div class="info-plain"><div class="label">מידע המטופל</div><div class="value">מסווג</div></div>
<div class="info-card"><div class="label">מקום ביצוע</div><div class="value">{location}</div></div>
<div class="pill {pill_class}">{pill_text}</div></div>"""


def _section_heading(text: str) -> str:
    return f'<div class="section-heading">{text}</div>'


def _results_table(headers: list[str], rows: list[list[str]]) -> str:
    # Cells are listed in visual right-to-left reading order (the first item is the
    # rightmost column); an RTL flex/table lays the first DOM child at the right, so
    # no reversal is needed here.
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _kv_table(pairs: list[tuple[str, str]]) -> str:
    # k (the label, e.g. a document name) sits on the right, v (its status) on the left.
    rows = "".join(f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>' for k, v in pairs)
    return f'<table class="kv-table">{rows}</table>'


def _field_row(right: tuple[str, str], left: tuple[str, str]) -> str:
    # The first DOM child of an RTL flex row renders on the right (confirmed by
    # measuring word coordinates against the originals with PyMuPDF).
    return f"""<div class="field-row">
<div class="field-card"><div class="label">{right[0]}</div><div class="value">{right[1]}</div></div>
<div class="field-card"><div class="label">{left[0]}</div><div class="value">{left[1]}</div></div>
</div>"""


def _note(text: str, variant: str = "") -> str:
    cls = f"note {variant}".strip()
    return f'<div class="{cls}">{text}</div>'


def _footer(right: str, left: str) -> str:
    return f'<div class="footer"><div>{right}</div><div>{left}</div></div></div>'


def _ecg_chart() -> str:
    return """<div class="chart-box"><svg viewBox="0 0 900 200" xmlns="http://www.w3.org/2000/svg">
<defs><pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
<path d="M 20 0 L 0 0 0 20" fill="none" stroke="#cfe6ef" stroke-width="1"/></pattern></defs>
<rect width="900" height="200" fill="url(#grid)"/>
<polyline points="0,100 60,100 90,100 105,40 120,175 135,60 165,100 210,100 225,100 240,55
255,150 270,80 300,100 345,100 360,100 375,40 390,175 405,60 435,100 480,100 495,100 510,55
525,150 540,80 570,100 615,100 630,100 645,40 660,175 675,60 705,100 750,100 765,100 780,55
795,150 810,80 840,100 900,100" fill="none" stroke="#0e5b66" stroke-width="2.6"
stroke-linecap="round" stroke-linejoin="round"/></svg></div>"""


def _signatures(right_label: str, left_label: str) -> str:
    return f"""<div class="sign-row">
<div class="sign-box"><div class="sign-line"></div><div class="label">{right_label}</div></div>
<div class="sign-box"><div class="sign-line"></div><div class="label">{left_label}</div></div>
</div>"""


def _cbc(issue_date: str) -> str:
    return (
        _hospital_header(issue_date)
        + _title_card("תוצאות מעבדה - ספירת דם מלאה", "מסמך תוצאות מעבדה")
        + _info_row("מסמך רפואי", "med")
        + _section_heading("תוצאות ספירת דם מלאה")
        + _results_table(
            ["בדיקה", "תוצאה", "טווח ייחוס", "יחידות"],
            [
                ["תאי דם לבנים", "6.8", "4.0 - 10.0", "אלפים למיקרוליטר"],
                ["תאי דם אדומים", "4.9", "4.5 - 5.9", "מיליונים למיקרוליטר"],
                ["המוגלובין", "14.2", "13.5 - 17.5", "גרם לדציליטר"],
                ["המטוקריט", "42.1", "41 - 53", "אחוזים"],
                ["נפח כדורית ממוצע", "86", "80 - 100", "פמטוליטר"],
                ["טסיות דם", "245", "150 - 400", "אלפים למיקרוליטר"],
            ],
        )
        + _note("הערה: לא זוהו חריגות משמעותיות בערכים המוצגים במסמך.")
        + _footer("המרכז הרפואי רמון", "מסמך ממוחשב")
    )


def _coagulation(issue_date: str) -> str:
    return (
        _hospital_header(issue_date)
        + _title_card("תוצאות מעבדה - בדיקות קרישה", "מסמך תוצאות מעבדה")
        + _info_row("מסמך רפואי", "med")
        + _section_heading("תוצאות בדיקות קרישה")
        + _results_table(
            ["בדיקה", "תוצאה", "טווח ייחוס", "יחידות"],
            [
                ["זמן פרותרומבין", "12.4", "11.0 - 14.0", "שניות"],
                ["יחס מנורמל בינלאומי", "1.02", "0.90 - 1.20", "יחס"],
                ["זמן תרומבופלסטין חלקי מופעל", "29.8", "25 - 35", "שניות"],
            ],
        )
        + _note("סיכום מעבדתי: ערכי הקרישה במסמך נמצאים בטווחי הייחוס המוצגים.")
        + _footer("המרכז הרפואי רמון", "מסמך ממוחשב")
    )


def _ecg(issue_date: str, exam_date: str) -> str:
    return (
        _hospital_header(issue_date)
        + _title_card("תרשים פעילות חשמלית של הלב", "בדיקת לב במנוחה")
        + _info_row("מסמך רפואי", "med")
        + _field_row(("קצב לב", "72 פעימות בדקה"), ("סוג הבדיקה", "תרשים פעילות חשמלית של הלב במנוחה"))
        + _field_row(("מועד ביצוע", exam_date), ("קצב", "סדיר"))
        + _ecg_chart()
        + _note("פענוח מסכם: תרשים סדיר ללא ממצא חריג בולט במסמך הדוגמה.")
        + _footer("המרכז הרפואי רמון", "מסמך ממוחשב")
    )


def _urinalysis(issue_date: str) -> str:
    return (
        _hospital_header(issue_date)
        + _title_card("תוצאות מעבדה - בדיקת שתן כללית", "מסמך תוצאות מעבדה")
        + _info_row("מסמך רפואי", "med")
        + _section_heading("תוצאות בדיקת שתן כללית")
        + _results_table(
            ["מדד", "תוצאה", "טווח ייחוס"],
            [
                ["משקל סגולי", "1.018", "1.005 - 1.030"],
                ["חומציות", "6.0", "5.0 - 8.0"],
                ["חלבון", "שלילי", "שלילי"],
                ["גלוקוז", "שלילי", "שלילי"],
                ["דם", "שלילי", "שלילי"],
                ["לויקוציטים", "שלילי", "שלילי"],
            ],
        )
        + _note(
            "זהו מסמך רפואי תקין מסוג בדיקת שתן. הוא מיועד לשמש בדמו כמסמך רפואי שאינו בהכרח "
            "רלוונטי לדרישות טרום-ניתוח שנבחרו.",
            "warn",
        )
        + _footer("המרכז הרפואי רמון", "מסמך ממוחשב")
    )


def _preop_summary(issue_date: str, visit_date: str) -> str:
    return (
        _hospital_header(issue_date)
        + _title_card("סיכום הערכה טרום-ניתוחית", "מסמך סיכום רפואי")
        + _info_row("מסמך רפואי", "med")
        + _section_heading("סיכום הערכה רפואית טרום-ניתוחית")
        + _field_row(("מחלקה", "מרפאה טרום-ניתוחית"), ("מטרת הביקור", "הערכת מוכנות לקראת הליך ניתוחי מתוכנן"))
        + _field_row(("מצב המסמך", "הושלם"), ("תאריך הביקור", visit_date))
        + _section_heading("מסמכים שנבדקו")
        + _kv_table(
            [
                ("ספירת דם מלאה", "התקבל"),
                ("בדיקות קרישה", "התקבל"),
                ("תרשים פעילות חשמלית של הלב", "התקבל"),
            ]
        )
        + _section_heading("הערות")
        + _note(
            "המסמכים המפורטים לעיל התקבלו ונקלטו בתיק. המשך התהליך כפוף להחלטת הצוות הרפואי "
            "ולנהלי בית החולים."
        )
        + _signatures("חותמת המחלקה", "חתימת גורם רפואי")
        + _footer("המרכז הרפואי רמון", "מסמך ממוחשב")
    )


def _electricity_bill(issue_date: str, bill_date: str, due_date: str, period: str) -> str:
    return (
        f"""<div class="topbar"></div><div class="page">
<div class="issue-date" style="text-align:left;">תאריך הפקה: {issue_date}</div>
{_title_card("חשבון חשמל", "מסמך שאינו רפואי")}
<div class="company-row"><div class="company-name"><div class="name">חברת החשמל האזורית</div>
<div class="sub">חשבון צריכת חשמל תקופתי</div></div>
<div class="company-meta"><div>תאריך חשבון: {bill_date}</div><div>מספר מסמך: 402781</div></div></div>
<div class="info-row">
<div class="info-plain"><div class="label">פרטי הלקוח</div><div class="value">מסווג</div></div>
<div class="info-card"><div class="label">סוג המסמך</div><div class="value">חשבון שירות</div></div>
<div class="pill non">לא מסמך רפואי</div></div>"""
        + _field_row(("צריכה", "486 קילוואט שעה"), ("תקופת חיוב", period))
        + f"""<div class="amount-card"><div class="label">סכום לתשלום</div>
<div class="value">₪ 384.70</div><div class="due">לתשלום עד {due_date}</div></div>"""
        + _section_heading("פירוט חיובים")
        + _kv_table(
            [
                ("צריכת חשמל", "₪ 286.10"),
                ("תשלום קבוע", "₪ 31.60"),
                ("מסים ותוספות", "₪ 67.00"),
            ]
        )
        + _footer("מסמך דוגמה", "הופק במערכת ממוחשבת")
    )


DOCUMENTS = {
    "cbc": ("01_ספירת_דם_מלאה", lambda d: _cbc(d["issue"])),
    "coagulation": ("02_בדיקות_קרישה", lambda d: _coagulation(d["issue"])),
    "ecg": ("03_תרשים_לב", lambda d: _ecg(d["issue"], d["ecg_exam"])),
    "urinalysis": ("04_בדיקת_שתן_לא_רלוונטית", lambda d: _urinalysis(d["issue"])),
    "preop_summary": ("05_סיכום_טרום_ניתוח", lambda d: _preop_summary(d["issue"], d["preop_visit"])),
    "electricity_bill": (
        "06_מסמך_לא_רפואי_חשבון_חשמל",
        lambda d: _electricity_bill(d["issue"], d["bill_date"], d["bill_due"], d["bill_period"]),
    ),
}


def _dates_for(issue_date: str) -> dict:
    """Every non-issue date, derived from the issue date's year so a different
    --date argument still produces internally consistent documents."""
    year = issue_date.split(".")[-1]
    return {
        "issue": issue_date,
        "ecg_exam": f"13.09.{year}",
        "preop_visit": f"14.09.{year}",
        "bill_date": f"08.09.{year}",
        "bill_due": f"28.09.{year}",
        "bill_period": f"יולי - אוגוסט {year}",
    }


def render(issue_date: str, out_dirs: list[Path]) -> None:
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    dates = _dates_for(issue_date)
    with tempfile.TemporaryDirectory() as tmp:
        for name, (hebrew_name, body_fn) in DOCUMENTS.items():
            body = body_fn(dates)
            html = (
                "<!doctype html><html lang='he' dir='rtl'><head><meta charset='utf-8'>"
                f"<style>{STYLE}</style></head><body>{body}</body></html>"
            )
            source = Path(tmp) / f"{name}.html"
            source.write_text(html, encoding="utf-8")
            target = Path(tmp) / f"{name}.pdf"
            subprocess.run(
                [
                    str(CHROME),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={target}",
                    source.as_uri(),
                ],
                check=True,
                capture_output=True,
            )
            for d in out_dirs:
                file_name = f"{hebrew_name}.pdf" if d.name != "2026" or d.parent.name != "fixtures" else f"{name}.pdf"
                (d / file_name).write_bytes(target.read_bytes())


if __name__ == "__main__":
    # The paths printed below are Hebrew; a Windows console is often cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    date_arg = sys.argv[1] if len(sys.argv) > 1 else "15.09.2026"
    outs = [Path(sys.argv[2])] if len(sys.argv) > 2 else [ROOT / "demo" / "2026", ROOT / "tests" / "fixtures" / "2026"]
    render(date_arg, outs)
    print(f"rendered {len(DOCUMENTS)} documents dated {date_arg} into", ", ".join(str(o) for o in outs))
