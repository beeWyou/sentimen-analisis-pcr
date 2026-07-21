from flask import Flask, render_template, request, redirect, url_for, flash, session
import joblib
import pandas as pd
import pymysql

app = Flask(__name__)
app.secret_key = "sentiment_pcr"

model = joblib.load("naive_bayes_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

USERNAME = "admin"
PASSWORD = "admin123"

db = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="db_sentimen"
)

cursor = db.cursor()

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == USERNAME and password == PASSWORD:
            return redirect(url_for("dashboard"))
        else:
            flash("Username atau Password salah!")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():

    tahun_filter = request.args.get("tahun")

    cursor = db.cursor(pymysql.cursors.DictCursor)

    # Ambil daftar tahun

    cursor.execute("""
        SELECT DISTINCT tahun
        FROM hasil_sentimen
        WHERE tahun IS NOT NULL
        ORDER BY tahun
    """)

    tahun_list = [row["tahun"] for row in cursor.fetchall()]

    # Query sesuai filter
    if tahun_filter:

        cursor.execute("""
            SELECT sentimen
            FROM hasil_sentimen
            WHERE tahun=%s
        """, (tahun_filter,))

    else:

        cursor.execute("""
            SELECT sentimen
            FROM hasil_sentimen
        """)

    data = cursor.fetchall()

    positif = sum(1 for row in data if row["sentimen"] == "Positif")
    netral  = sum(1 for row in data if row["sentimen"] == "Netral")
    negatif = sum(1 for row in data if row["sentimen"] == "Negatif")

    return render_template(
        "dashboard.html",
        positif=positif,
        netral=netral,
        negatif=negatif,
        tahun_list=tahun_list,
        tahun_filter=tahun_filter
    )

def tidak_ada_saran(teks):
    teks = str(teks).strip().lower()

    daftar = [
        "", "-", "--", "---", ".", "..", "...", "....",
        "_", "__", "___", "n/a", "na", "null"
    ]

    if teks in daftar:
        return True

    if len(teks) == 1 and teks.isalpha():
        return True

    if teks.isdigit():
        return True

    return False


@app.route("/upload", methods=["GET", "POST"])
def upload():

    tahun_filter = request.args.get("tahun")

    # PAGINATION
    page = request.args.get("page", 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    # UPLOAD FILE
    if request.method == "POST":

        if "file" not in request.files:
            flash("Silakan pilih file terlebih dahulu.")
            return redirect(url_for("upload"))

        file = request.files["file"]

        if file.filename == "":
            flash("Silakan pilih file.")
            return redirect(url_for("upload"))

        try:

            df = pd.read_excel(file, header=5)
            df.columns = df.columns.str.strip()

            # Kolom Saran wajib ada
            if "Saran" not in df.columns:
                flash("Kolom Saran tidak ditemukan.")
                return redirect(url_for("upload"))

            # Kalau Periode tidak ada
            if "Periode" not in df.columns:
                df["Periode"] = None

            df = df[["Periode", "Saran"]].copy()

            df["Saran"] = (
                df["Saran"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            mask = df["Saran"].apply(tidak_ada_saran)
            tidak_ada = int(mask.sum())

            df = df[~mask].copy()

            if len(df) == 0:
                flash("Tidak ada saran yang dapat dianalisis.")
                return redirect(url_for("upload"))

            # ANALISIS SENTIMEN

            X = tfidf.transform(df["Saran"])
            df["sentimen"] = model.predict(X)

            cursor = db.cursor()

            for _, row in df.iterrows():

                try:
                    tahun = int(row["Periode"])
                except:
                    tahun = None

                cursor.execute("""
                    INSERT INTO hasil_sentimen
                    (saran, tahun, sentimen)
                    VALUES (%s, %s, %s)
                """, (
                    row["Saran"],
                    tahun,
                    row["sentimen"]
                ))

            db.commit()

            flash("Data berhasil dianalisis dan disimpan ke database.")

            return redirect(url_for("upload"))

        except Exception as e:

            flash(f"Terjadi kesalahan : {e}")

            return redirect(url_for("upload"))

    # AMBIL DATA

    cursor = db.cursor(pymysql.cursors.DictCursor)

    # Hitung total data
    if tahun_filter:

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM hasil_sentimen
            WHERE tahun=%s
        """, (tahun_filter,))

    else:

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM hasil_sentimen
        """)

    total_data = cursor.fetchone()["total"]
    total_page = (total_data + per_page - 1) // per_page

    # Ambil data sesuai halaman
    if tahun_filter:

        cursor.execute("""
            SELECT *
            FROM hasil_sentimen
            WHERE tahun=%s
            ORDER BY id DESC
            LIMIT %s OFFSET %s
        """, (tahun_filter, per_page, offset))

    else:

        cursor.execute("""
            SELECT *
            FROM hasil_sentimen
            ORDER BY id DESC
            LIMIT %s OFFSET %s
        """, (per_page, offset))

    data = cursor.fetchall()

    # LIST TAHUN

    cursor.execute("""
        SELECT DISTINCT tahun
        FROM hasil_sentimen
        WHERE tahun IS NOT NULL
        ORDER BY tahun ASC
    """)

    tahun_list = [row["tahun"] for row in cursor.fetchall()]

    # TOTAL CARD

    cursor.execute("""
        SELECT sentimen, COUNT(*) AS jumlah
        FROM hasil_sentimen
        GROUP BY sentimen
    """)

    hasil = cursor.fetchall()

    positif = 0
    netral = 0
    negatif = 0

    for row in hasil:

        if row["sentimen"] == "Positif":
            positif = row["jumlah"]

        elif row["sentimen"] == "Netral":
            netral = row["jumlah"]

        elif row["sentimen"] == "Negatif":
            negatif = row["jumlah"]

    tidak_ada = 0

    return render_template(
        "upload.html",
        data=data,
        positif=positif,
        netral=netral,
        negatif=negatif,
        tidak_ada=tidak_ada,
        tahun_list=tahun_list,
        tahun_filter=tahun_filter,
        page=page,
        total_page=total_page,
        per_page=per_page
    )
if __name__ == "__main__":
    app.run(debug=True)