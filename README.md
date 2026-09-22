# Document Service

שירות עצמאי, קטן, שמקבל מהמטופל קובצי PDF רפואיים, מסווג אותם באמצעות מודל שפה, שומר רק את
הקבצים שהתקבלו באחסון פרטי ב-S3, ומחזיר למי שמריץ בדיקה (Hospital Agent, דרך `CheckDocuments`)
את רשימת המסמכים של המטופל ותוצאת הקליטה של כל אחד. בנוי כמו `appointment-service`: container
יחיד ב-FastAPI, מסד SQLite פנימי למטא-דאטה בלבד (הקבצים עצמם ב-S3), אימות בכותרת `X-API-Key`,
ותשובות שגיאה כקודי JSON.

## מה השירות עושה

לכל קובץ שמועלה, השירות מריץ שרשרת בדיקות בסדר קבוע (עיצוב §4.2) - הכשל הראשון קובע את התוצאה,
וקובץ נשמר ב-S3 רק אם התוצאה `ACCEPTED`:

| קוד תוצאה | משמעות |
|---|---|
| `ACCEPTED` | קובץ PDF קריא, רפואי, מסוג מוכר בקטלוג (`CBC`, `COAGULATION_TESTS`, `ECG`, `URINALYSIS`, `PREOP_SUMMARY`), שייך למטופל שמעלה, לא הועלה כבר בעבר, ובתוך תקופת התוקף של סוגו. נשמר ב-S3. |
| `NON_MEDICAL_DOCUMENT` | הטקסט חולץ בהצלחה, אבל המסווג קבע שהמסמך אינו רפואי (חשבון, מכתב וכדומה). |
| `DOCUMENT_UNREADABLE` | לא PDF תקין (לא מתחיל ב-`%PDF-`, גדול מ-10MB), אין בו שכבת טקסט (סריקה בלי OCR), חורג מ-20 עמודים או מ-200,000 תווי טקסט, קריאת המסווג נכשלה או לא החזירה תשובה שמישה, המסמך רפואי אך לא מסוג מוכר, או שתאריך ההפקה שלו עתידי. כל אלה כשל סגור - "לא קריא" ולא ניחוש. |
| `DOCUMENT_EXPIRED` | אין תאריך הפקה במסמך (נחשב פג-תוקף - כשל סגור), או שהתאריך ישן יותר מתוקף הסוג (90 יום ל-CBC/קרישה/שתן, 180 ל-ECG, 30 לסיכום טרום-ניתוח). |
| `DUPLICATE_DOCUMENT` | אותו קובץ בדיוק (גיבוב SHA-256 זהה) כבר התקבל בעבר (`ACCEPTED`) עבור אותו מטופל. קובץ שנדחה בעבר נבדק מחדש בהעלאה חוזרת - כשל סיווג חד-פעמי לא נועל אותו לצמיתות. |
| `PATIENT_MISMATCH` | המסמך מציין מזהה מטופל אחר בתבנית ה-IdP (`P-<ספרות>`) שאינו המטופל שמעלה. מזהה שאינו בתבנית הזו (מספר מסמך, מספר לקוח וכו') מתעלמים ממנו. |

רק מסמך `ACCEPTED` נשמר ב-S3, במפתח `patients/<patient_id>/<document_id>.pdf`; מסמך שנדחה
נשמר רק כשורת מטא-דאטה (תוצאה, גיבוב, גודל) - לעולם לא התוכן עצמו.

## הפעלה

דרישה מקדימה: Docker Desktop פעיל.

יש להעתיק את `.env.example` אל `.env` ולמלא את הערכים: `DOCUMENT_API_KEY` (מחרוזת אקראית),
`OPENAI_API_KEY` ו-`S3_BUCKET` עם פרטי ה-IAM (ראו שני הסעיפים הבאים). קובץ `.env` מוחרג מ-Git.

```bash
docker compose up --build -d
docker compose ps
```

השירות מאזין ל-`127.0.0.1` בלבד (לא לרשת החיצונית), על הפורט 8090:

- Health: `http://localhost:8090/health`
- Swagger: `http://localhost:8090/docs`

בלי `OPENAI_API_KEY` או בלי `S3_BUCKET`, ה-health יחזיר `classifier`/`storage` בתור
`not_configured`, והשירות עדיין עולה - אבל כל העלאה נענית ב-`503 {"error": "service_not_configured"}`
עד שהם מוגדרים.

### מגבלות ההעלאה

קובץ מועלה מוגבל ל-10MB ולכל היותר 20 עמודים (200,000 תווי טקסט מחולץ); חריגה מכל אחד מהם
מתקבלת כ-`DOCUMENT_UNREADABLE`. לפני שהניתוב עצמו רץ, middleware ב-ASGI טהור
(`AuthBeforeBodyMiddleware` ב-`app/main.py`) בודק כל בקשה שמתחילה ב-`/api/` על סמך הכותרות
בלבד, בלי לקרוא שורת גוף אחת - כך שקובץ ענק או בקשה לא מאומתת נדחים לפני שהם נקראים בכלל
לזיכרון או לדיסק (ב-FastAPI, ניתוח multipart-form קורה בשכבת הניתוב, לפני שקוד ה-route רץ):

- מפתח `X-API-Key` חסר או שגוי -> `401 {"error": "unauthorized"}`.
- בקשת `POST` בלי כותרת `Content-Length` -> `411 {"error": "length_required"}`.
- בקשת `POST` עם `Content-Length` שגדול מ-10MB + 64KB (תקורת multipart) -> `413 {"error": "too_large"}`.

הניתוב עצמו בודק את הגודל גם בפועל (קורא לכל היותר גודל המגבלה + בית אחד), כהגנת עומק.

## הקמת S3 ו-IAM (פעם אחת, ב-AWS Console)

השלבים הבאים מבוצעים ידנית בקונסולת AWS, פעם אחת לכל סביבה. שום סוד לא נקרא, לא מודפס ולא
מועתק על ידי קוד השירות - רק המפתחות שנוצרים בסוף נכנסים ל-`.env`.

1. **S3 -> Create bucket.** שם ייחודי, למשל `hospital-agent-documents-<סיומת אקראית>`, אזור
   `eu-north-1` (כמו מופע ה-RDS של Hospital Agent). *Block all public access* - מופעל
   (ברירת המחדל). הצפנת ברירת מחדל - SSE-S3 (`AES-256`). גרסאות (Versioning) - אופציונלי.

2. **Permissions -> Bucket policy** - מדיניות שדוחה כל גישה שאינה מוצפנת (TLS), עם `BUCKET`
   מוחלף בשם הדלי שנוצר:

   ```json
   {"Version":"2012-10-17","Statement":[{"Sid":"DenyInsecureTransport","Effect":"Deny","Principal":"*",
     "Action":"s3:*","Resource":["arn:aws:s3:::BUCKET","arn:aws:s3:::BUCKET/*"],
     "Condition":{"Bool":{"aws:SecureTransport":"false"}}}]}
   ```

3. **IAM -> Users -> Create user**, בשם `document-service`, בלי גישת קונסול (No console
   access). מצרפים לו מדיניות inline (least privilege - רק על תיקיית `patients/` בדלי):

   ```json
   {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:PutObject","s3:GetObject"],
     "Resource":"arn:aws:s3:::BUCKET/patients/*"}]}
   ```

4. **המשתמש -> Security credentials -> Create access key**, use case: *Application running
   outside AWS*. את `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` ואת שם הדלי (`S3_BUCKET`)
   מכניסים ל-`.env` בלבד - לעולם לא לקוד, ל-image או ל-Git.

## מפתח OpenAI

`OPENAI_API_KEY` נכנס ל-`.env`. הקריאה היחידה למודל היא לסיווג מסמך (§4.3 בעיצוב): הטקסט
שחולץ מה-PDF, מוגבל ל-8,000 התווים הראשונים, נשלח יחד עם prompt קבוע שמבקש אך ורק סיווג -
האם רפואי, איזה סוג מהקטלוג (אם בכלל), תאריך ההפקה, ומזהה מטופל אם מופיע במפורש. הפרומפט אוסר
במפורש פרשנות של תוצאות בדיקה או ייעוץ רפואי. תשובה שלא עומדת בסכימה, או קריאה שנכשלת, מתקבלת
כ-`DOCUMENT_UNREADABLE` (כשל סגור) - שום דבר לא מנוסה שוב באופן אוטומטי.

## API

שתי נקודות קצה, שתיהן דורשות את הכותרת `X-API-Key`:

**העלאת מסמך:**

```bash
curl -i -X POST \
  -H "X-API-Key: <DOCUMENT_API_KEY>" \
  -F "file=@cbc.pdf;type=application/pdf" \
  http://localhost:8090/api/v1/patients/P-10041/documents
```

תשובה (`201`):

```json
{"document_id": "DOC-3F2A1B9C0D4E", "document_type": "CBC", "document_date": "2026-09-15", "result": "ACCEPTED"}
```

**רשימת המסמכים של מטופל:**

```bash
curl -i -H "X-API-Key: <DOCUMENT_API_KEY>" http://localhost:8090/api/v1/patients/P-10041/documents
```

תשובה (`200`):

```json
{"documents": [{"document_id": "DOC-3F2A1B9C0D4E", "document_type": "CBC", "document_date": "2026-09-15",
                "result": "ACCEPTED", "uploaded_at": "2026-09-22T10:15:00+00:00"}]}
```

`document_type` ו-`document_date` הם `null` כשאין להם ערך ודאי (למשל `NON_MEDICAL_DOCUMENT`
או `DOCUMENT_UNREADABLE` ללא תאריך). הרשימה ממוינת לפי זמן ההעלאה, כולל מסמכים שנדחו.

## קבצי הדמו

שישה קובצי PDF להדגמה חיה, ותיאור התוצאה הצפויה מכל אחד, נמצאים ב-`demo/README.md`
(הקבצים עצמם ב-`demo/2026/`, בעברית; המקבילים באנגלית לבדיקות האוטומטיות ב-`tests/fixtures/`).

## בדיקות

הבדיקות הרגילות אינן פונות לרשת בכלל (`tests/conftest.py` מנקה את משתני הסביבה החיים לפני כל
בדיקה), ורצות בתוך container נקי:

```bash
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/src" -w /src \
  -e PIP_ROOT_USER_ACTION=ignore -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  python:3.12-slim sh -c "pip install -q -r requirements-dev.txt && pytest -q -p no:cacheprovider"
```

`tests/test_live.py` הן הבדיקות היחידות שפונות לרשת - קריאה אמיתית ל-OpenAI וכתיבה אמיתית
לדלי - ולכן מדולגות כברירת מחדל. כדי להריץ אותן, מתוך שורש הפרויקט, ב-Git Bash, כשה-`.env`
כבר מכיל `OPENAI_API_KEY` ו-`S3_BUCKET` אמיתיים:

```bash
MSYS_NO_PATHCONV=1 docker run --rm --env-file .env -v "$(pwd -W):/src" -w /src python:3.12-slim \
  sh -c 'pip install -q -r requirements-dev.txt && LIVE_OPENAI_API_KEY="$OPENAI_API_KEY" \
  LIVE_S3_BUCKET="$S3_BUCKET" RUN_LIVE_LLM=1 RUN_LIVE_S3=1 pytest -q -p no:cacheprovider tests/test_live.py'
```

`--env-file .env` מעביר את המשתנים אל תוך ה-container בלבד; שום סוד לא מודפס לפלט.

## פרטיות

רק מסמך שהתקבל (`ACCEPTED`) נשמר בפועל - ב-S3 בלבד, פרטי ומוצפן, לעולם לא בכתובת ציבורית.
מסמך שנדחה שומר רק שורת מטא-דאטה: תוצאה, גיבוב SHA-256 וגודל - לא תוכנו, ולא שם הקובץ.
יומן ההרצה (`stdout`, JSON) אינו מכיל לעולם מזהה מטופל או שם קובץ. ה-audit (הטבלה
`document_audit_logs`) הוא הרשומה היחידה שמחזיקה מזהה מטופל, ומחזיקה חוץ מזה רק קודים
ומזהים - פעולה, תוצאה, מזהה מסמך וזמן תגובה - לעולם לא טקסט, שם קובץ או תוכן.

## עצירה וניקוי

```bash
docker compose down
```

למחיקת נתוני SQLite המקומיים בלבד (לא ה-S3):

```bash
docker compose down -v
```
