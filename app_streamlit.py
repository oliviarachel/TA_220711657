# ============================================================
# STREAMLIT APP
# ANALISIS SENTIMEN MULTI-LABEL ULASAN AVOSKIN
# SVM BASELINE
# ============================================================

import os
import io
import re
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.special import expit, softmax
from sklearn.preprocessing import MultiLabelBinarizer


# ============================================================
# 1. KONFIGURASI STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Analisis Sentimen Avoskin",
    page_icon="💬",
    layout="wide"
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. LOKASI FILE MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "svm_model_baseline.pkl"
)

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "preprocessor_tfidf_baseline.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "threshold_per_label_baseline.pkl"
)

CLASSES_PATH = os.path.join(
    BASE_DIR,
    "classes_baseline.pkl"
)


# ============================================================
# 3. CEK FILE MODEL
# ============================================================

required_files = {
    "Model SVM": MODEL_PATH,
    "TF-IDF": TFIDF_PATH,
    "Threshold": THRESHOLD_PATH,
    "Classes": CLASSES_PATH
}

missing_files = [
    name
    for name, path in required_files.items()
    if not os.path.exists(path)
]

if missing_files:

    st.error(
        "File model berikut tidak ditemukan:\n\n"
        + "\n".join(
            f"- {name}: {required_files[name]}"
            for name in missing_files
        )
    )

    st.stop()


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # --------------------------------------------------------
    # Load SVM
    # --------------------------------------------------------

    svm_model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Load TF-IDF
    # --------------------------------------------------------

    tfidf = joblib.load(
        TFIDF_PATH
    )

    # --------------------------------------------------------
    # Load classes
    # --------------------------------------------------------

    with open(
        CLASSES_PATH,
        "rb"
    ) as f:

        classes = pickle.load(f)

    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    thresholds = joblib.load(
        THRESHOLD_PATH
    )

    return (
        svm_model,
        tfidf,
        classes,
        thresholds
    )


# ============================================================
# 5. MEMUAT MODEL
# ============================================================

with st.spinner("Memuat model..."):

    try:

        (
            svm_model,
            tfidf,
            classes,
            thresholds
        ) = load_model()

    except Exception as e:

        st.error(
            f"Gagal memuat model:\n\n{e}"
        )

        st.stop()


# ============================================================
# 6. PROSES CLASSES
# ============================================================

if isinstance(
    classes,
    MultiLabelBinarizer
):

    class_names = list(
        classes.classes_
    )

else:

    class_names = list(
        classes
    )


# Pastikan threshold berbentuk array

thresholds = np.asarray(
    thresholds,
    dtype=float
)


# ============================================================
# 7. VALIDASI MODEL
# ============================================================

if len(thresholds) != len(class_names):

    st.error(
        "Jumlah threshold tidak sama "
        "dengan jumlah label."
    )

    st.stop()


if len(svm_model.estimators_) != len(class_names):

    st.error(
        "Jumlah estimator SVM tidak sama "
        "dengan jumlah label."
    )

    st.stop()


# ============================================================
# 8. JUDUL APLIKASI
# ============================================================

st.title(
    "Analisis Sentimen Multi-Label Ulasan Avoskin"
)

st.write(
    "Aplikasi ini menggunakan model "
    "**Support Vector Machine (SVM)** "
    "untuk melakukan klasifikasi sentimen "
    "multi-label berbasis aspek."
)


# ============================================================
# 9. INFORMASI MODEL
# ============================================================

with st.expander(
    "Informasi Model",
    expanded=False
):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            "SVM"
        )

    with col2:

        st.metric(
            "Jumlah Label",
            len(class_names)
        )

    with col3:

        st.metric(
            "Threshold",
            "Per Label"
        )

    st.write(
        "**Label yang digunakan:**"
    )

    for i, label in enumerate(
        class_names,
        start=1
    ):

        st.write(
            f"{i}. {label} "
            f"(threshold = {thresholds[i-1]:.3f})"
        )


# ============================================================
# 10. FUNGSI CLEAN TEXT
# ============================================================

def clean_text(text):

    """
    Membersihkan teks sebelum masuk ke TF-IDF.
    """

    if pd.isna(text):

        return ""

    text = str(text)

    # lowercase

    text = text.lower()

    # hapus spasi berlebih

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # hapus spasi awal dan akhir

    text = text.strip()

    return text


# ============================================================
# 11. FUNGSI MENENTUKAN KOLOM TEKS
# ============================================================

TEXT_COLUMN = None


def find_text_column(df):

    # --------------------------------------------------------
    # Jika kolom ditentukan secara manual
    # --------------------------------------------------------

    if TEXT_COLUMN is not None:

        if TEXT_COLUMN not in df.columns:

            raise ValueError(
                f"Kolom '{TEXT_COLUMN}' tidak ditemukan.\n"
                f"Kolom tersedia: {list(df.columns)}"
            )

        return TEXT_COLUMN


    # --------------------------------------------------------
    # Kandidat nama kolom
    # --------------------------------------------------------

    candidates = [
        "text",
        "ulasan",
        "review",
        "komentar",
        "comment",
        "content",
        "teks",
        "review_text",
        "review_text_clean",
        "clean_text"
    ]


    # --------------------------------------------------------
    # Cek nama kolom
    # --------------------------------------------------------

    for col in candidates:

        if col in df.columns:

            return col


    # --------------------------------------------------------
    # Cari kolom bertipe object/string
    # --------------------------------------------------------

    text_columns = df.select_dtypes(
        include=[
            "object",
            "string"
        ]
    ).columns.tolist()


    if len(text_columns) == 1:

        return text_columns[0]


    raise ValueError(
        "Kolom teks tidak dapat ditentukan "
        "secara otomatis.\n\n"
        f"Kolom yang tersedia:\n"
        f"{list(df.columns)}\n\n"
        "Silakan tentukan TEXT_COLUMN "
        "secara manual."
    )


# ============================================================
# 12. FUNGSI PREDIKSI MULTI-LABEL
# ============================================================

def predict_multilabel(
    df,
    text_column
):

    # --------------------------------------------------------
    # Ambil teks
    # --------------------------------------------------------

    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
    )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    texts_clean = texts.apply(
        clean_text
    )


    # --------------------------------------------------------
    # Transform TF-IDF
    # --------------------------------------------------------

    X = tfidf.transform(
        texts_clean
    )


    # --------------------------------------------------------
    # Decision function SVM
    # --------------------------------------------------------

    decision_scores = (
        svm_model.decision_function(X)
    )


    # --------------------------------------------------------
    # Jika hanya satu label
    # --------------------------------------------------------

    if decision_scores.ndim == 1:

        decision_scores = (
            decision_scores.reshape(
                -1,
                1
            )
        )


    # ========================================================
    # 12A. CONFIDENCE SCORE MODEL
    # ========================================================
    #
    # Decision score SVM diubah ke rentang 0-1 menggunakan
    # sigmoid/expit.
    #
    # Nilai ini digunakan untuk:
    # 1. Confidence Level
    # 2. Threshold per-label
    #
    # Confidence setiap label dihitung secara independen.
    # Oleh karena itu jumlahnya TIDAK harus 100%.
    # ========================================================

    confidence_scores = expit(
        decision_scores
    )


    # ========================================================
    # 12B. PROBABILITAS RELATIF ANTAR LABEL
    # ========================================================
    #
    # Softmax digunakan untuk menormalisasi skor antar label.
    #
    # Untuk setiap satu ulasan:
    # jumlah probabilitas seluruh 12 label = 100%.
    #
    # Nilai ini hanya digunakan sebagai probabilitas relatif
    # antar label pada tampilan aplikasi.
    # ========================================================

    probabilities = softmax(
        decision_scores,
        axis=1
    )


    # ========================================================
    # 12C. TERAPKAN THRESHOLD PER-LABEL
    # ========================================================
    #
    # Threshold tetap diterapkan pada confidence_scores,
    # bukan pada probabilitas Softmax.
    #
    # Dengan demikian threshold per-label dari penelitian
    # tetap digunakan.
    # ========================================================

    predictions = (
        confidence_scores
        >= thresholds.reshape(
            1,
            -1
        )
    ).astype(int)


    # --------------------------------------------------------
    # Buat dataframe hasil
    # --------------------------------------------------------

    result = df.copy()


    # ========================================================
    # LABEL HASIL
    # ========================================================

    label_results = []


    for row in predictions:

        labels = [
            class_names[i]
            for i, value in enumerate(row)
            if value == 1
        ]


        # Jika tidak ada label

        if len(labels) == 0:

            labels = [
                "tidak_terdeteksi"
            ]


        label_results.append(
            ", ".join(labels)
        )


    result[
        "predicted_labels"
    ] = label_results


    # ========================================================
    # PROBABILITAS SETIAP LABEL
    # ========================================================

    for i, label in enumerate(
        class_names
    ):

        result[
            f"probability_{label}"
        ] = probabilities[:, i]


    # ========================================================
    # CONFIDENCE LEVEL SETIAP LABEL
    # ========================================================

    for i, label in enumerate(
        class_names
    ):

        result[
            f"confidence_{label}"
        ] = confidence_scores[:, i]


    # ========================================================
    # LABEL DENGAN PROBABILITAS TERTINGGI
    # ========================================================

    highest_label_index = np.argmax(
        probabilities,
        axis=1
    )


    result[
        "top_label"
    ] = [
        class_names[i]
        for i in highest_label_index
    ]


    result[
        "top_probability"
    ] = [
        probabilities[
            row,
            highest_label_index[row]
        ]
        for row in range(
            len(result)
        )
    ]


    # ========================================================
    # CONFIDENCE LABEL TERATAS
    # ========================================================

    result[
        "top_confidence"
    ] = [
        confidence_scores[
            row,
            highest_label_index[row]
        ]
        for row in range(
            len(result)
        )
    ]


    return (
        result,
        probabilities,
        predictions,
        confidence_scores
    )


# ============================================================
# 13. PILIH MODE INPUT
# ============================================================

st.header("Input Ulasan")

input_mode = st.radio(
    "Pilih metode input:",
    [
        "Single Ulasan",
        "Upload File"
    ],
    horizontal=True
)


# ============================================================
# 14. MODE SINGLE ULASAN
# ============================================================

if input_mode == "Single Ulasan":

    st.write(
        "Masukkan satu ulasan untuk mendapatkan "
        "prediksi multi-label."
    )


    single_text = st.text_area(
        "Masukkan ulasan:",
        placeholder=(
            "Contoh: Produknya bagus dan packaging-nya aman, "
            "tapi harganya agak mahal."
        ),
        height=150
    )


    if st.button(
        "🔍 Analisis Ulasan",
        type="primary"
    ):

        if not single_text.strip():

            st.warning(
                "Silakan masukkan ulasan terlebih dahulu."
            )

        else:

            # ------------------------------------------------
            # Buat dataframe satu baris
            # ------------------------------------------------

            df_single = pd.DataFrame({
                "ulasan": [
                    single_text
                ]
            })


            with st.spinner(
                "Sedang melakukan prediksi..."
            ):

                try:

                    (
                        result_df,
                        probabilities,
                        predictions,
                        confidence_scores
                    ) = predict_multilabel(
                        df_single,
                        "ulasan"
                    )

                except Exception as e:

                    st.error(
                        f"Terjadi kesalahan saat prediksi:\n\n{e}"
                    )

                    st.stop()


            st.success(
                "Prediksi selesai!"
            )


            # =================================================
            # HASIL PREDIKSI
            # =================================================

            st.subheader(
                "Hasil Prediksi"
            )


            predicted_labels = result_df[
                "predicted_labels"
            ].iloc[0]


            top_label = result_df[
                "top_label"
            ].iloc[0]


            top_probability = result_df[
                "top_probability"
            ].iloc[0] * 100


            st.write(
                f"**Ulasan:** {single_text}"
            )


            st.write(
                f"**Label Prediksi:** "
                f"{predicted_labels}"
            )


            st.write(
                f"**Label dengan Probabilitas Tertinggi:** "
                f"{top_label}"
            )


            # =================================================
            # LABEL TERPILIH + CONFIDENCE LEVEL
            # =================================================

            selected_labels = []


            for i, label in enumerate(
                class_names
            ):

                if predictions[0, i] == 1:

                    confidence = (
                        confidence_scores[0, i]
                        * 100
                    )

                    selected_labels.append(
                        f"{label} "
                        f"({confidence:.2f}%)"
                    )


            if len(selected_labels) == 0:

                confidence_text = (
                    "Tidak ada label yang memenuhi threshold."
                )

            else:

                confidence_text = (
                    ", ".join(
                        selected_labels
                    )
                )


            st.write(
                f"**Confidence Level:** "
                f"{confidence_text}"
            )


            # =================================================
            # PROBABILITAS SETIAP LABEL
            # =================================================

            st.subheader(
                "Probabilitas Setiap Label"
            )


            probability_data = []


            for i, label in enumerate(
                class_names
            ):

                probability = (
                    probabilities[0, i]
                    * 100
                )


                selected = (
                    "Terpilih"
                    if predictions[0, i] == 1
                    else "Tidak terpilih"
                )


                probability_data.append({

                    "Label": label,

                    "Probabilitas (%)": (
                        f"{probability:.2f}%"
                    ),

                    "Prediksi": selected
                })


            probability_df = pd.DataFrame(
                probability_data
            )


            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # Total probabilitas
            # ------------------------------------------------

            total_probability = (
                probabilities[0].sum()
                * 100
            )


            st.caption(
                f"Total probabilitas: "
                f"{total_probability:.2f}%"
            )


            # =================================================
            # CONFIDENCE LEVEL LABEL TERPILIH
            # =================================================

            st.subheader(
                "Confidence Level Label Terpilih"
            )


            confidence_data = []


            for i, label in enumerate(
                class_names
            ):

                if predictions[0, i] == 1:

                    confidence_data.append({

                        "Label": label,

                        "Confidence Level (%)": (
                            f"{confidence_scores[0, i] * 100:.2f}%"
                        ),

                        "Threshold (%)": (
                            f"{thresholds[i] * 100:.2f}%"
                        )
                    })


            if len(confidence_data) > 0:

                confidence_df = pd.DataFrame(
                    confidence_data
                )


                st.dataframe(
                    confidence_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "Tidak ada label yang memenuhi threshold."
                )


# ============================================================
# 15. MODE UPLOAD FILE
# ============================================================

else:

    st.write(
        "Upload file **.xlsx** atau **.csv** "
        "yang berisi beberapa ulasan untuk diprediksi."
    )


    uploaded_file = st.file_uploader(
        "Pilih file data",
        type=[
            "xlsx",
            "csv"
        ]
    )


    if uploaded_file is not None:

        filename = uploaded_file.name


        # ----------------------------------------------------
        # Baca XLSX
        # ----------------------------------------------------

        if filename.lower().endswith(
            ".xlsx"
        ):

            try:

                df_input = pd.read_excel(
                    uploaded_file
                )

            except Exception as e:

                st.error(
                    f"Gagal membaca file Excel: {e}"
                )

                st.stop()


        # ----------------------------------------------------
        # Baca CSV
        # ----------------------------------------------------

        elif filename.lower().endswith(
            ".csv"
        ):

            try:

                df_input = pd.read_csv(
                    uploaded_file,
                    encoding="utf-8"
                )

            except UnicodeDecodeError:

                df_input = pd.read_csv(
                    uploaded_file,
                    encoding="latin1"
                )

            except Exception as e:

                st.error(
                    f"Gagal membaca file CSV: {e}"
                )

                st.stop()


        else:

            st.error(
                "Format file harus .xlsx atau .csv"
            )

            st.stop()


        # ====================================================
        # 16. INFORMASI DATA
        # ====================================================

        st.success(
            f"File berhasil dibaca: {filename}"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Jumlah Baris",
                len(df_input)
            )


        with col2:

            st.metric(
                "Jumlah Kolom",
                len(df_input.columns)
            )


        # ====================================================
        # 17. TENTUKAN KOLOM TEKS
        # ====================================================

        try:

            text_column = find_text_column(
                df_input
            )

        except ValueError as e:

            st.error(
                str(e)
            )

            st.stop()


        st.info(
            f"Kolom teks yang digunakan: "
            f"**{text_column}**"
        )


        # ====================================================
        # 18. PREVIEW DATA
        # ====================================================

        st.subheader(
            "Preview Data"
        )


        st.dataframe(
            df_input.head(10),
            use_container_width=True
        )


        # ====================================================
        # 19. TOMBOL PREDIKSI
        # ====================================================

        if st.button(
            "🔍 Mulai Prediksi",
            type="primary"
        ):

            with st.spinner(
                "Sedang melakukan prediksi..."
            ):

                try:

                    (
                        result_df,
                        probabilities,
                        predictions,
                        confidence_scores
                    ) = predict_multilabel(
                        df_input,
                        text_column
                    )

                except Exception as e:

                    st.error(
                        f"Terjadi kesalahan saat prediksi:\n\n{e}"
                    )

                    st.stop()


            st.success(
                "Prediksi selesai!"
            )


            # =================================================
            # 20. HASIL PREDIKSI
            # =================================================

            st.header(
                "Hasil Prediksi"
            )


            display_result = result_df[
                [
                    text_column,
                    "predicted_labels",
                    "top_label",
                    "top_probability"
                ]
            ].copy()


            display_result[
                "top_probability"
            ] = (
                display_result[
                    "top_probability"
                ] * 100
            ).round(2)


            display_result = (
                display_result.rename(
                    columns={
                        "predicted_labels":
                            "Label Prediksi",

                        "top_label":
                            "Label Teratas",

                        "top_probability":
                            "Probabilitas Teratas (%)"
                    }
                )
            )


            st.dataframe(
                display_result,
                use_container_width=True,
                height=500
            )


            # =================================================
            # 21. PROBABILITAS SETIAP LABEL
            # =================================================

            st.header(
                "Probabilitas Setiap Label"
            )


            probability_display = result_df[
                [text_column]
                + [
                    f"probability_{label}"
                    for label in class_names
                ]
            ].copy()


            # Ubah ke persen

            for label in class_names:

                probability_display[
                    f"probability_{label}"
                ] = (
                    probability_display[
                        f"probability_{label}"
                    ] * 100
                ).round(2)


            # Rename kolom

            probability_display = (
                probability_display.rename(
                    columns={
                        f"probability_{label}":
                        f"{label} (%)"
                        for label in class_names
                    }
                )
            )


            st.dataframe(
                probability_display,
                use_container_width=True,
                height=500
            )


            # =================================================
            # 22. TOTAL PROBABILITAS
            # =================================================

            probability_columns = [
                f"probability_{label}"
                for label in class_names
            ]


            total_probability_each_row = (
                result_df[
                    probability_columns
                ].sum(axis=1)
                * 100
            )


            st.write(
                "**Pemeriksaan total probabilitas:**"
            )


            total_probability_display = pd.DataFrame({

                text_column:
                    result_df[
                        text_column
                    ],

                "Total Probabilitas (%)":
                    total_probability_each_row.round(2)

            })


            st.dataframe(
                total_probability_display,
                use_container_width=True
            )


            # =================================================
            # 23. LABEL TERPILIH + CONFIDENCE LEVEL
            # =================================================

            st.header(
                "Label Terpilih dan Confidence Level"
            )


            confidence_results = []


            for row_idx in range(
                len(result_df)
            ):

                for label_idx, label in enumerate(
                    class_names
                ):

                    if predictions[
                        row_idx,
                        label_idx
                    ] == 1:

                        confidence_results.append({

                            text_column:
                                result_df[
                                    text_column
                                ].iloc[row_idx],

                            "Label":
                                label,

                            "Confidence Level (%)":
                                round(
                                    confidence_scores[
                                        row_idx,
                                        label_idx
                                    ] * 100,
                                    2
                                ),

                            "Threshold (%)":
                                round(
                                    thresholds[
                                        label_idx
                                    ] * 100,
                                    2
                                )
                        })


            if len(confidence_results) > 0:

                confidence_display = pd.DataFrame(
                    confidence_results
                )


                st.dataframe(
                    confidence_display,
                    use_container_width=True,
                    height=500
                )

            else:

                st.info(
                    "Tidak ada label yang memenuhi threshold."
                )


            # =================================================
            # 24. DOWNLOAD HASIL
            # =================================================

            st.header(
                "Download Hasil"
            )


            # -------------------------------------------------
            # Tambahkan probabilitas ke hasil download
            # -------------------------------------------------

            result_percentage = (
                result_df.copy()
            )


            for label in class_names:

                result_percentage[
                    f"probability_{label}"
                ] = (
                    result_percentage[
                        f"probability_{label}"
                    ] * 100
                ).round(2)


            # -------------------------------------------------
            # Tambahkan confidence level
            # -------------------------------------------------

            for label in class_names:

                result_percentage[
                    f"confidence_{label}"
                ] = (
                    result_percentage[
                        f"confidence_{label}"
                    ] * 100
                ).round(2)


            # -------------------------------------------------
            # Confidence level hanya untuk label terpilih
            # -------------------------------------------------

            label_confidence_results = []


            for row_idx in range(
                len(result_df)
            ):

                current_labels = []


                for label_idx, label in enumerate(
                    class_names
                ):

                    if predictions[
                        row_idx,
                        label_idx
                    ] == 1:

                        confidence = (
                            confidence_scores[
                                row_idx,
                                label_idx
                            ] * 100
                        )


                        current_labels.append(
                            f"{label} "
                            f"({confidence:.2f}%)"
                        )


                if len(current_labels) == 0:

                    current_labels.append(
                        "tidak_terdeteksi"
                    )


                label_confidence_results.append(
                    ", ".join(
                        current_labels
                    )
                )


            result_percentage[
                "label_dan_confidence"
            ] = label_confidence_results


            # -------------------------------------------------
            # Excel
            # -------------------------------------------------

            excel_buffer = io.BytesIO()


            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                result_percentage.to_excel(
                    writer,
                    index=False,
                    sheet_name="Hasil Prediksi"
                )


            excel_buffer.seek(0)


            st.download_button(
                label="📥 Download Hasil Excel",
                data=excel_buffer,
                file_name="hasil_prediksi_multilabel.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )


            # -------------------------------------------------
            # CSV
            # -------------------------------------------------

            csv_data = (
                result_percentage
                .to_csv(
                    index=False,
                    encoding="utf-8-sig"
                )
            )


            st.download_button(
                label="📥 Download Hasil CSV",
                data=csv_data,
                file_name="hasil_prediksi_multilabel.csv",
                mime="text/csv"
            )


# ============================================================
# 25. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Analisis Sentimen Multi-Label Ulasan Avoskin "
    "menggunakan SVM"
)
