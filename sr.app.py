import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from itertools import combinations
from collections import Counter

st.set_page_config(
    page_title="Sistem Rekomendasi Buku",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Sistem Rekomendasi Buku")
st.markdown(
    "Aplikasi rekomendasi buku berdasarkan **pola kemunculan bersama "
    "(item-based co-occurrence)** dari interaksi pengguna."
)

st.sidebar.header("Pengaturan Analisis")
st.sidebar.caption(
    "Unggah tiga dataset Book-Crossing dalam format CSV. "
    "File dibaca menggunakan pemisah titik koma (;)."
)

@st.cache_data(show_spinner=False)
def read_dataset(file, dataset_name):
    if dataset_name == "books":
        return pd.read_csv(
            file, sep=";", encoding="latin-1",
            engine="python", on_bad_lines="warn"
        )
    return pd.read_csv(
        file, sep=";", encoding="latin-1",
        na_values=["NULL"], low_memory=False
    )

@st.cache_data(show_spinner=False)
def prepare_data(users, ratings, books):
    users = users.copy()
    ratings = ratings.copy()
    books = books.copy()

    required = {
        "users": ["User-ID", "Location", "Age"],
        "ratings": ["User-ID", "ISBN", "Book-Rating"],
        "books": ["ISBN", "Book-Title", "Book-Author"]
    }
    for name, columns in required.items():
        frame = {"users": users, "ratings": ratings, "books": books}[name]
        missing = [col for col in columns if col not in frame.columns]
        if missing:
            raise ValueError(
                f"Kolom wajib pada dataset {name} tidak ditemukan: {missing}"
            )

    users["Age"] = pd.to_numeric(users["Age"], errors="coerce")
    ratings["Book-Rating"] = pd.to_numeric(
        ratings["Book-Rating"], errors="coerce"
    )
    if "Year-Of-Publication" in books.columns:
        books["Year-Of-Publication"] = pd.to_numeric(
            books["Year-Of-Publication"], errors="coerce"
        )

    books_unique = books.drop_duplicates(subset="ISBN", keep="first")
    explicit = ratings[
        ratings["Book-Rating"].between(1, 10)
    ].copy()

    interaction = ratings[["User-ID", "ISBN"]].dropna(
        subset=["User-ID", "ISBN"]
    ).drop_duplicates(subset=["User-ID", "ISBN"])

    return users, ratings, books, books_unique, explicit, interaction

@st.cache_data(show_spinner=False)
def build_recommendation_data(interaction, books_unique,
                              min_book_users, max_books_per_user):
    book_frequency = interaction.groupby("ISBN")["User-ID"].nunique()
    valid_isbn = book_frequency[
        book_frequency >= min_book_users
    ].index

    filtered = interaction[
        interaction["ISBN"].isin(valid_isbn)
    ].copy()

    user_frequency = filtered.groupby("User-ID")["ISBN"].nunique()
    valid_users = user_frequency[
        user_frequency <= max_books_per_user
    ].index
    filtered = filtered[
        filtered["User-ID"].isin(valid_users)
    ].copy()

    user_books = (
        filtered.groupby("User-ID")["ISBN"]
        .apply(lambda values: sorted(set(values)))
    )

    pair_counter = Counter()
    for isbn_list in user_books:
        pair_counter.update(combinations(isbn_list, 2))

    cooccurrence = pd.DataFrame(
        [
            (isbn_a, isbn_b, count)
            for (isbn_a, isbn_b), count in pair_counter.items()
        ],
        columns=["ISBN_A", "ISBN_B", "Cooccurrence"]
    )

    if cooccurrence.empty:
        return filtered, cooccurrence

    book_info = books_unique[
        ["ISBN", "Book-Title", "Book-Author"]
    ].drop_duplicates(subset="ISBN")

    cooccurrence = cooccurrence.merge(
        book_info.rename(columns={
            "ISBN": "ISBN_A",
            "Book-Title": "Book_A",
            "Book-Author": "Author_A"
        }),
        on="ISBN_A", how="left"
    )
    cooccurrence = cooccurrence.merge(
        book_info.rename(columns={
            "ISBN": "ISBN_B",
            "Book-Title": "Book_B",
            "Book-Author": "Author_B"
        }),
        on="ISBN_B", how="left"
    )

    total_users = filtered["User-ID"].nunique()
    item_count = filtered.groupby("ISBN")["User-ID"].nunique()

    cooccurrence["Count_A"] = cooccurrence["ISBN_A"].map(item_count)
    cooccurrence["Count_B"] = cooccurrence["ISBN_B"].map(item_count)
    cooccurrence["Support"] = (
        cooccurrence["Cooccurrence"] / total_users
    )
    cooccurrence["Confidence_A_B"] = (
        cooccurrence["Cooccurrence"] / cooccurrence["Count_A"]
    )
    cooccurrence["Confidence_B_A"] = (
        cooccurrence["Cooccurrence"] / cooccurrence["Count_B"]
    )
    cooccurrence["Lift"] = (
        cooccurrence["Cooccurrence"] * total_users
        / (cooccurrence["Count_A"] * cooccurrence["Count_B"])
    )
    cooccurrence["Jaccard"] = (
        cooccurrence["Cooccurrence"]
        / (
            cooccurrence["Count_A"] + cooccurrence["Count_B"]
            - cooccurrence["Cooccurrence"]
        )
    )
    cooccurrence["Cosine"] = (
        cooccurrence["Cooccurrence"]
        / np.sqrt(cooccurrence["Count_A"] * cooccurrence["Count_B"])
    )

    cooccurrence = cooccurrence.sort_values(
        "Cooccurrence", ascending=False
    ).reset_index(drop=True)

    return filtered, cooccurrence

def get_recommendations(isbn, cooccurrence, top_n, min_cooccurrence):
    if cooccurrence.empty:
        return pd.DataFrame()

    result = cooccurrence[
        (cooccurrence["ISBN_A"] == isbn)
        | (cooccurrence["ISBN_B"] == isbn)
    ].copy()

    if result.empty:
        return pd.DataFrame()

    is_a = result["ISBN_A"] == isbn
    result["ISBN_Rekomendasi"] = np.where(
        is_a, result["ISBN_B"], result["ISBN_A"]
    )
    result["Judul Buku"] = np.where(
        is_a, result["Book_B"], result["Book_A"]
    )
    result["Penulis"] = np.where(
        is_a, result["Author_B"], result["Author_A"]
    )
    result["Confidence"] = np.where(
        is_a, result["Confidence_A_B"], result["Confidence_B_A"]
    )

    result = result[
        result["Cooccurrence"] >= min_cooccurrence
    ].sort_values(
        ["Lift", "Cooccurrence"], ascending=False
    )

    return result[[
        "ISBN_Rekomendasi", "Judul Buku", "Penulis",
        "Cooccurrence", "Confidence", "Lift", "Jaccard", "Cosine"
    ]].head(top_n)

# Unggah dataset
col1, col2, col3 = st.columns(3)
with col1:
    users_file = st.file_uploader(
        "Dataset Pengguna (BX-Users.csv)", type=["csv"], key="users"
    )
with col2:
    ratings_file = st.file_uploader(
        "Dataset Rating (BX-Book-Ratings.csv)", type=["csv"], key="ratings"
    )
with col3:
    books_file = st.file_uploader(
        "Dataset Buku (BX-Books.csv)", type=["csv"], key="books"
    )

if not all([users_file, ratings_file, books_file]):
    st.info(
        "Silakan unggah ketiga file CSV untuk memulai analisis. "
        "Pastikan file berasal dari dataset Book-Crossing."
    )
    st.markdown("**Kolom utama yang diperlukan**")
    st.write(
        "- Pengguna: `User-ID`, `Location`, `Age`\n"
        "- Rating: `User-ID`, `ISBN`, `Book-Rating`\n"
        "- Buku: `ISBN`, `Book-Title`, `Book-Author`"
    )
    st.stop()

try:
    with st.spinner("Membaca dan menyiapkan dataset..."):
        users_raw = read_dataset(users_file, "users")
        ratings_raw = read_dataset(ratings_file, "ratings")
        books_raw = read_dataset(books_file, "books")
        users, ratings, books, books_unique, explicit, interaction = (
            prepare_data(users_raw, ratings_raw, books_raw)
        )
except Exception as error:
    st.error(f"Dataset gagal dibaca: {error}")
    st.stop()

st.sidebar.subheader("Parameter Rekomendasi")
min_book_users = st.sidebar.number_input(
    "Minimum pengguna per buku", min_value=1, value=20, step=1
)
max_books_per_user = st.sidebar.number_input(
    "Maksimum buku per pengguna", min_value=1, value=30, step=1
)
min_cooccurrence = st.sidebar.number_input(
    "Minimum kemunculan bersama", min_value=1, value=5, step=1
)
top_n = st.sidebar.number_input(
    "Jumlah rekomendasi", min_value=1, max_value=50, value=10, step=1
)

with st.spinner("Menghitung pasangan buku yang sering muncul bersama..."):
    filtered_interaction, cooccurrence = build_recommendation_data(
        interaction, books_unique,
        int(min_book_users), int(max_books_per_user)
    )

if filtered_interaction.empty or cooccurrence.empty:
    st.warning(
        "Belum ditemukan pasangan buku dengan parameter tersebut. "
        "Coba turunkan minimum pengguna per buku atau sesuaikan "
        "maksimum buku per pengguna."
    )
    st.stop()

tab_overview, tab_eksplorasi, tab_pasangan, tab_rekomendasi = st.tabs([
    "Ringkasan", "Eksplorasi Data", "Pasangan Buku", "Rekomendasi"
])

with tab_overview:
    st.subheader("Ringkasan Dataset")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pengguna", f"{users['User-ID'].nunique():,}")
    m2.metric("Buku", f"{books_unique['ISBN'].nunique():,}")
    m3.metric("Total interaksi", f"{len(ratings):,}")
    m4.metric("Rating eksplisit", f"{len(explicit):,}")

    m5, m6, m7 = st.columns(3)
    m5.metric("Interaksi setelah pembersihan", f"{len(interaction):,}")
    m6.metric("Interaksi setelah filtering", f"{len(filtered_interaction):,}")
    m7.metric("Pasangan buku terbentuk", f"{len(cooccurrence):,}")

    st.markdown("#### Kualitas Data")
    quality = pd.DataFrame([
        {
            "Dataset": "Pengguna",
            "Jumlah baris": len(users),
            "Jumlah kolom": users.shape[1],
            "Missing values": int(users.isna().sum().sum()),
            "Duplikasi baris": int(users.duplicated().sum())
        },
        {
            "Dataset": "Rating",
            "Jumlah baris": len(ratings),
            "Jumlah kolom": ratings.shape[1],
            "Missing values": int(ratings.isna().sum().sum()),
            "Duplikasi baris": int(ratings.duplicated().sum())
        },
        {
            "Dataset": "Buku",
            "Jumlah baris": len(books),
            "Jumlah kolom": books.shape[1],
            "Missing values": int(books.isna().sum().sum()),
            "Duplikasi baris": int(books.duplicated().sum())
        }
    ])
    st.dataframe(quality, use_container_width=True, hide_index=True)

    st.markdown("#### Tahapan Pembentukan Rekomendasi")
    st.write(
        "1. Mengambil pasangan unik pengguna dan ISBN.\n"
        "2. Memilih buku yang dibaca minimal sesuai parameter.\n"
        "3. Mempertahankan pengguna dengan jumlah buku tidak melebihi batas.\n"
        "4. Menghitung pasangan buku yang muncul pada pengguna yang sama.\n"
        "5. Menghitung Support, Confidence, Lift, Jaccard, dan Cosine Similarity."
    )

with tab_eksplorasi:
    st.subheader("Eksplorasi Data")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Distribusi Rating Buku**")
        rating_dist = (
            ratings["Book-Rating"].value_counts()
            .sort_index().rename_axis("Rating").reset_index(name="Jumlah")
        )
        fig = px.bar(
            rating_dist, x="Rating", y="Jumlah",
            labels={"Rating": "Rating buku", "Jumlah": "Jumlah interaksi"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Kelompok Usia Pengguna**")
        age_bins = [4, 17, 24, 34, 44, 54, 64, 100]
        age_labels = ["≤17", "18–24", "25–34", "35–44",
                      "45–54", "55–64", "65+"]
        age_group = pd.cut(
            users["Age"].where(users["Age"].between(5, 100)),
            bins=age_bins, labels=age_labels
        ).value_counts().sort_index().rename_axis(
            "Kelompok usia"
        ).reset_index(name="Jumlah pengguna")
        fig = px.bar(
            age_group, x="Kelompok usia", y="Jumlah pengguna",
            labels={"Kelompok usia": "Kelompok usia"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Buku dengan Rating Eksplisit Terbanyak**")
    popular = (
        explicit.groupby("ISBN")
        .agg(Jumlah_Rating=("Book-Rating", "count"),
             Rata_Rating=("Book-Rating", "mean"))
        .reset_index()
        .merge(
            books_unique[["ISBN", "Book-Title", "Book-Author"]],
            on="ISBN", how="left"
        )
        .sort_values("Jumlah_Rating", ascending=False)
        .head(15)
    )
    st.dataframe(
        popular[["Book-Title", "Book-Author", "Jumlah_Rating", "Rata_Rating"]],
        use_container_width=True, hide_index=True
    )

with tab_pasangan:
    st.subheader("Pasangan Buku yang Sering Dibaca Bersama")
    top_pairs = cooccurrence.head(15).copy()
    top_pairs["Pasangan Buku"] = (
        top_pairs["Book_A"].fillna(top_pairs["ISBN_A"].astype(str))
        + " & "
        + top_pairs["Book_B"].fillna(top_pairs["ISBN_B"].astype(str))
    )
    fig = px.bar(
        top_pairs.sort_values("Cooccurrence"),
        x="Cooccurrence", y="Pasangan Buku", orientation="h",
        labels={
            "Cooccurrence": "Jumlah pengguna yang sama",
            "Pasangan Buku": "Pasangan buku"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    display_pairs = cooccurrence[[
        "Book_A", "Author_A", "Book_B", "Author_B",
        "Cooccurrence", "Support", "Confidence_A_B",
        "Confidence_B_A", "Lift", "Jaccard", "Cosine"
    ]].head(100).rename(columns={
        "Book_A": "Judul Buku A",
        "Author_A": "Penulis A",
        "Book_B": "Judul Buku B",
        "Author_B": "Penulis B",
        "Cooccurrence": "Kemunculan Bersama",
        "Support": "Support",
        "Confidence_A_B": "Confidence A ke B",
        "Confidence_B_A": "Confidence B ke A",
        "Lift": "Lift",
        "Jaccard": "Jaccard",
        "Cosine": "Cosine Similarity"
    })
    st.dataframe(display_pairs, use_container_width=True, hide_index=True)
    st.download_button(
        "Unduh seluruh hasil pasangan buku (CSV)",
        data=cooccurrence.to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_pasangan_buku.csv",
        mime="text/csv"
    )

with tab_rekomendasi:
    st.subheader("Cari Rekomendasi Buku")
    book_catalog = books_unique[["ISBN", "Book-Title", "Book-Author"]].copy()
    book_catalog["Book-Title"] = book_catalog["Book-Title"].fillna(
        "Judul tidak tersedia"
    )
    book_catalog["Book-Author"] = book_catalog["Book-Author"].fillna(
        "Penulis tidak tersedia"
    )
    book_catalog["Pilihan"] = (
        book_catalog["Book-Title"] + " — " + book_catalog["Book-Author"]
        + " [" + book_catalog["ISBN"].astype(str) + "]"
    )
    book_catalog = book_catalog.drop_duplicates("ISBN").sort_values("Book-Title")

    search_text = st.text_input(
        "Cari judul buku, penulis, atau ISBN",
        placeholder="Ketik kata kunci buku..."
    )
    if search_text.strip():
        mask = (
            book_catalog["Pilihan"].str.contains(
                search_text.strip(), case=False, na=False, regex=False
            )
        )
        book_options = book_catalog.loc[mask].head(100)
    else:
        book_options = book_catalog.head(100)

    if book_options.empty:
        st.warning("Buku tidak ditemukan. Coba kata kunci lain.")
        st.stop()

    selected_label = st.selectbox(
        "Pilih buku dari hasil pencarian",
        options=book_options["Pilihan"].tolist()
    )
    selected_isbn = book_options.loc[
        book_options["Pilihan"] == selected_label, "ISBN"
    ].iloc[0]

    selected_row = book_catalog[
        book_catalog["ISBN"] == selected_isbn
    ].iloc[0]
    st.markdown(f"**Buku pilihan:** {selected_row['Book-Title']}")
    st.caption(f"Penulis: {selected_row['Book-Author']} | ISBN: {selected_isbn}")

    recommendations = get_recommendations(
        selected_isbn, cooccurrence, int(top_n), int(min_cooccurrence)
    )
    if recommendations.empty:
        st.warning(
            "Belum ada rekomendasi untuk buku ini berdasarkan parameter "
            "filter yang digunakan. Coba pilih buku lain atau turunkan "
            "minimum kemunculan bersama."
        )
    else:
        st.success(f"Ditemukan {len(recommendations)} rekomendasi buku.")
        st.dataframe(
            recommendations, use_container_width=True, hide_index=True
        )
        st.download_button(
            "Unduh rekomendasi (CSV)",
            data=recommendations.to_csv(index=False).encode("utf-8-sig"),
            file_name="rekomendasi_buku.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "Metode rekomendasi: item-based co-occurrence. "
    "Rating 0 tetap dihitung sebagai interaksi pada pembentukan pasangan buku, "
    "sesuai proses pada notebook; rating eksplisit 1–10 digunakan untuk "
    "eksplorasi rating."
)
