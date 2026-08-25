# ============================================================
# Protein–İlaç Bağlanma Afinitesi Tahmini — İnteraktif Dashboard
# ============================================================
# Çalıştırmak için terminalde (venv aktifken):
#   pip install streamlit plotly joblib
#   streamlit run app.py
# (Bu dosyayı indirdikten sonra proje klasöründe adını app.py yap.)
#
# ÖNEMLİ: Bu sürüm, Random Forest modelini bulutta eğitmek yerine
# önceden eğitilmiş 'model/rf_model.pkl' dosyasını yükler (bkz.
# train_model.py). Bu dosyayı önce train_model.py ile oluşturup
# GitHub'a eklemen gerekiyor, yoksa uygulama hata verir.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ------------------------------------------------------------
# Sayfa ayarları ve renk temamız (notebook/rapordakiyle aynı)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Protein–İlaç Bağlanma Afinitesi",
    page_icon="🧬",
    layout="wide",
)

PETROL = "#0A4D47"
PETROL_DARK = "#063430"
MINT = "#8FD9C4"
LIGHT_BG = "#DCEEE9"
PASTEL_MURDUM = "#EAD9E4"
MURDUM_TEXT = "#4A2F44"
# Vurgu (accent) rengi — raporun korelasyon ısı haritasında zaten kullanılan
# turuncu tonu, göze çarpması istenen yerlerde (metrikler, butonlar, aktif
# sekme çizgisi) kullanılıyor. Petrol/mint temayla uyumlu ama daha canlı.
ACCENT = "#D9822B"
ACCENT_DARK = "#B96C1B"
PETROL_SCALE = [[0, MINT], [1, PETROL]]
TURUNCU_PETROL_SCALE = [[0, "#D9822B"], [0.5, "#FFFFFF"], [1, "#0B6B57"]]

# CSS: metrik kutuları, başlıklar, expander, tablo başlığı, vurgu rengi
# (metrik değerleri, aktif sekme çizgisi, butonlar, uyarı kutuları), koyu
# petrol degradeli "profesyonel" kenar çubuğu ve sağ alt köşede belli
# belirsiz nefes alan bir nokta animasyonu.
st.markdown(f"""
<style>
[data-testid="stMetric"] {{
    background-color: {LIGHT_BG};
    border-radius: 10px;
    padding: 12px 8px;
    border: 1px solid {PETROL}22;
    border-left: 4px solid {ACCENT};
}}
[data-testid="stMetricValue"] {{
    color: {ACCENT} !important;
}}
h1, h2, h3 {{
    color: {PETROL};
}}
[data-testid="stExpander"] summary {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {PETROL};
}}
table thead tr th {{
    background-color: {PASTEL_MURDUM} !important;
    color: {MURDUM_TEXT} !important;
    font-weight: 600 !important;
}}

/* Aktif sekmenin altındaki çizgi ve seçili sekme metni vurgu renginde */
[data-baseweb="tab-highlight"] {{
    background-color: {ACCENT} !important;
}}
[data-baseweb="tab-list"] button[aria-selected="true"] p {{
    color: {ACCENT} !important;
    font-weight: 700 !important;
}}

/* İndirme butonu vurgu renginde */
[data-testid="stDownloadButton"] button {{
    background-color: {ACCENT} !important;
    color: white !important;
    border: none !important;
}}
[data-testid="stDownloadButton"] button:hover {{
    background-color: {ACCENT_DARK} !important;
    color: white !important;
}}

/* Bilgi/başarı/uyarı kutularının sol kenarına ince bir vurgu çizgisi */
[data-testid="stAlert"] {{
    border-left: 5px solid {ACCENT} !important;
    border-radius: 6px;
}}

/* ---------- Kenar çubuğu: koyu petrol degrade, "profesyonel" panel ---------- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {PETROL} 0%, {PETROL_DARK} 100%);
}}
[data-testid="stSidebar"] * {{
    color: #EAF6F2 !important;
}}
[data-testid="stSidebar"] h2 {{
    color: #FFFFFF !important;
    font-weight: 800 !important;
    letter-spacing: 0.2px;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.18) !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background-color: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 10px;
    margin-bottom: 8px;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    color: {MINT} !important;
    font-size: 1.02rem;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
    color: #FFFFFF !important;
}}
[data-testid="stSidebar"] [data-baseweb="tag"] {{
    background-color: {ACCENT} !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    background-color: rgba(255,255,255,0.94) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] input {{
    color: #222 !important;
}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: {MINT} !important;
    opacity: 0.85;
}}

@keyframes nefes-al {{
    0%   {{ transform: scale(1);    opacity: 0.35; }}
    50%  {{ transform: scale(1.4);  opacity: 0.75; }}
    100% {{ transform: scale(1);    opacity: 0.35; }}
}}
.nefes-noktasi {{
    position: fixed;
    bottom: 22px;
    right: 22px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: {ACCENT};
    animation: nefes-al 4s ease-in-out infinite;
    z-index: 9999;
    pointer-events: none;
}}
</style>
<div class="nefes-noktasi"></div>
""", unsafe_allow_html=True)

features = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]

# Bellek dostu veri tipleri: Streamlit Cloud'un ücretsiz katmanında bellek
# sınırlı (1GB). float32/int16 kullanmak float64/int64'e göre belleği yarıya
# indiriyor.
_DTYPES = {
    "protein": "category",
    "pKi": "float32",
    "MW": "float32",
    "LogP": "float32",
    "TPSA": "float32",
    "HBD": "int16",
    "HBA": "int16",
    "RotBonds": "int16",
}


@st.cache_data
def veri_yukle():
    # Sadece uygulamada gerçekten kullanılan sütunlar okunuyor — SMILES ve
    # source sütunları (uzun metin) belleği gereksiz yere şişirdiği için
    # dahil edilmiyor.
    return pd.read_csv(
        "data/combined_clean.csv.gz",
        usecols=list(_DTYPES.keys()),
        dtype=_DTYPES,
    )


@st.cache_resource
def model_yukle():
    # Model bulutta eğitilmiyor — train_model.py ile önceden eğitilip
    # kaydedilmiş dosya doğrudan yükleniyor. Bu, en büyük bellek/CPU
    # yükünü (fit() işlemini) buluttan tamamen kaldırıyor.
    return joblib.load("model/rf_model.pkl")


# ------------------------------------------------------------
# Grafik üretimi önbelleğe alınıyor: her widget etkileşiminde (örn. Tahmin
# Aracı'ndaki bir slider'ı oynatmak) Streamlit TÜM sekmelerin kodunu yeniden
# çalıştırıyor. Önbellek olmadan bu, ilgisiz sekmelerdeki ağır grafikleri de
# her seferinde yeniden oluşturup gereksiz bellek/CPU tüketimine yol açıyordu.
# ------------------------------------------------------------
@st.cache_data
def yap_histogram(_df):
    counts, bin_edges = np.histogram(_df["pKi"], bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    fig = go.Figure(go.Bar(
        x=bin_centers, y=counts, width=bin_width * 0.95,
        marker=dict(color=counts, colorscale=PETROL_SCALE),
        hovertemplate="pKi ≈ %{x:.2f}<br>Molekül sayısı: %{y:,}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="pKi", yaxis_title="Molekül sayısı", height=380,
                       margin=dict(t=10, b=10), hovermode="closest")
    return fig


@st.cache_data
def yap_boxplot(_df, secilen: tuple):
    subset = _df[_df["protein"].isin(secilen)]
    fig = px.box(
        subset, x="pKi", y="protein", color="protein",
        points=False,  # Aykırı değer noktaları çizilmiyor — binlerce nokta belleği şişiriyordu.
        color_discrete_sequence=px.colors.sample_colorscale([MINT, PETROL], len(secilen)),
    )
    fig.update_layout(showlegend=False, height=420, margin=dict(t=10, b=10),
                       yaxis_title="", xaxis_title="pKi")
    return fig


@st.cache_data
def yap_scatter(_df, x_degisken: str, y_degisken: str):
    ornek = _df.sample(min(8000, len(_df)), random_state=42)
    fig = px.scatter(
        ornek, x=x_degisken, y=y_degisken, color="pKi",
        color_continuous_scale=PETROL_SCALE, opacity=0.6,
        hover_data={x_degisken: ":.2f", y_degisken: ":.2f", "pKi": ":.2f", "protein": True},
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(height=420, margin=dict(t=10, b=10), hovermode="closest")
    return fig


@st.cache_data
def yap_heatmap(_df):
    corr = _df[["pKi"] + features].corr(method="spearman")
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale=TURUNCU_PETROL_SCALE,
        color_continuous_midpoint=0, aspect="auto",
    )
    fig.update_layout(height=450, margin=dict(t=10, b=10))
    return fig


@st.cache_data
def yap_r2_bar(sonuc_tablosu):
    renkler = [ACCENT if model == "Random Forest" else PETROL for model in sonuc_tablosu["Model"]]
    fig = go.Figure(go.Bar(
        y=sonuc_tablosu["Model"], x=sonuc_tablosu["Test R²"], orientation="h",
        marker_color=renkler,
        hovertemplate="%{y}<br>Test R²: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Test R²",
                       height=320, margin=dict(t=10, b=10), hovermode="closest")
    return fig


@st.cache_data
def yap_feature_importance(_onem_ozellik: tuple, _onem_deger: tuple):
    fig = go.Figure(go.Bar(
        y=list(_onem_ozellik), x=list(_onem_deger), orientation="h",
        marker=dict(color=list(_onem_deger), colorscale=PETROL_SCALE),
        hovertemplate="%{y}<br>Önem: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Önem Puanı",
                       height=320, margin=dict(t=10, b=10), hovermode="closest")
    return fig


with st.spinner("Veri yükleniyor..."):
    df = veri_yukle()

# ------------------------------------------------------------
# Kenar Çubuğu (Sidebar) — proje bilgisi ve genel filtreler
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧬 Proje Hakkında")
    st.markdown(
        "Bu uygulama, kinaz ailesi proteinler için **bağlanma afinitesi (pKi)** "
        "tahminine yönelik veri bilimi projesinin interaktif özetidir."
    )

    with st.expander("📚 Veri Kaynakları"):
        st.markdown(
            "- **BindingDB** — Ki/IC50/Kd, kinaz filtreli\n"
            "- **Davis** — pKd (log ölçekli)\n"
            "- **KIBA** — toplandı, farklı ölçek nedeniyle nihai sete dahil edilmedi"
        )

    with st.expander("🧾 Yöntem Özeti"):
        st.markdown(
            "1. Veri temizleme ve pKi hesaplama\n"
            "2. RDKit ile 6 moleküler tanımlayıcı\n"
            "3. Keşifsel analiz ve hipotez testleri\n"
            "4. 6 makine öğrenmesi modeli\n"
            "5. Bu interaktif dashboard"
        )

    with st.expander("⚠️ Kısıtlamalar"):
        st.markdown(
            "- BindingDB verisi yalnızca hedef adında \"kinase\" geçen kayıtlarla filtrelendi; "
            "bulgular kinaz ailesiyle sınırlı, tüm proteinlere genellenemez.\n"
            "- Ki, IC50 ve Kd farklı deneysel ölçüm türleri; ortak pKi ölçeğine çevrilerek birleştirildi.\n"
            "- \"<\" / \">\" sansür işaretleri kaldırılıp değerler nokta tahmini olarak kullanıldı.\n"
            "- Davis'teki pKd=5.0 tabanı muhtemelen gerçek bir ölçüm değil, eşik altı bağlanmalar için "
            "atanan bir kongvansiyon.\n"
            "- KIBA veri seti toplandı ama farklı ölçek nedeniyle nihai sete dahil edilmedi.\n"
            "- ML modelleri yalnızca ligand tabanlı tanımlayıcıları kullandı, protein kimliği/yapısı "
            "modele dahil edilmedi — performansı sınırlayan en önemli etken.\n"
            "- Ortam kısıtları nedeniyle XGBoost yerine Gradient Boosting kullanıldı."
        )

    st.divider()
    st.markdown("**Protein filtresi** *(Keşifsel Analiz sekmesini etkiler)*")
    top_proteins_all = df["protein"].value_counts().head(15).index.tolist()
    secilen_proteinler = st.multiselect(
        "Proteinler:", options=top_proteins_all, default=top_proteins_all[:10],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Beyza Fatıma Çekerekli · 2026")

# ------------------------------------------------------------
# Üst başlık ve özet metrikler
# ------------------------------------------------------------
st.title("🧬 Protein–İlaç Bağlanma Afinitesi Tahmini")
st.caption("Kinaz ailesi proteinler için pKi tahmini — interaktif proje dashboard'u. Tüm grafiklerde fare ile üzerine gelerek değerleri görebilirsin.")
st.markdown(
    f"<span style='color:{PETROL}; font-weight:600;'>Yol haritası:</span> "
    f"<span style='color:#444;'>Veri &nbsp;→&nbsp; Keşifsel Analiz &nbsp;→&nbsp; "
    f"Hipotez Testleri &nbsp;→&nbsp; Makine Öğrenmesi &nbsp;→&nbsp; Tahmin Aracı &nbsp;→&nbsp; Sonuç</span>",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Kayıt", f"{len(df):,}")
col2.metric("Benzersiz Protein", f"{df['protein'].nunique():,}")
col3.metric("Ortalama pKi", f"{df['pKi'].mean():.2f}")
col4.metric("En Yüksek pKi", f"{df['pKi'].max():.2f}")

st.download_button(
    "⬇️ Temizlenmiş veriyi CSV olarak indir",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="combined_clean.csv",
    mime="text/csv",
)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Keşifsel Analiz", "🧪 Hipotez Testleri", "🤖 Makine Öğrenmesi", "🔮 Tahmin Aracı", "📌 Sonuç"]
)

# ============================================================
# TAB 1: Keşifsel Veri Analizi
# ============================================================
with tab1:
    st.markdown(
        "Bu sekmede veri setinin genel örüntülerini görsel olarak inceliyoruz — "
        "sol menüdeki protein filtresi ve aşağıdaki seçim kutularıyla grafikleri özelleştirebilirsin."
    )

    st.subheader("pKi Dağılımı")
    st.caption(
        "pKi=5.0 ve pKi=7.0 civarındaki yığılmalar veri toplama sürecinden kaynaklanan "
        "bilinen bir yapaylıktır (bkz. rapor, Bölüm 4.1). Barın üzerine gel, o aralıktaki "
        "tam molekül sayısını gör."
    )
    st.plotly_chart(yap_histogram(df), width="stretch")

    st.subheader("Protein Bazında pKi Dağılımı")
    st.caption("Farklı protein hedeflerinde pKi'nin medyan ve yayılımı belirgin şekilde değişiyor.")
    if secilen_proteinler:
        fig = yap_boxplot(df, tuple(sorted(secilen_proteinler)))
        # Bu grafik tamamen statik gösteriliyor: fare üzerine gelince hiçbir
        # hover kutusu, zoom, sürükleme veya imleç değişikliği olmayacak.
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "staticPlot": True,
                "displayModeBar": False,
                "scrollZoom": False,
                "doubleClick": False,
            },
        )
    else:
        st.info("Sol menüden en az bir protein seç.")

    st.subheader("Değişken İlişkisi")
    st.caption(
        "İstediğin iki değişkeni seçip aralarındaki ilişkiyi anında görebilirsin. "
        "Noktaların rengi pKi değerine göre değişiyor, üzerine gelince tüm değerleri görürsün."
    )
    col_a, col_b = st.columns(2)
    x_degisken = col_a.selectbox("X ekseni", features, index=1)
    y_degisken = col_b.selectbox("Y ekseni", ["pKi"] + features, index=0)
    st.plotly_chart(yap_scatter(df, x_degisken, y_degisken), width="stretch")

    st.subheader("Korelasyon Matrisi (Spearman)")
    st.caption(
        "pKi ile en güçlü ilişki MW, TPSA ve HBA'da; LogP ile ilişki neredeyse sıfır. "
        "Tanımlayıcıların kendi aralarında ise daha güçlü ilişkiler var — örneğin TPSA-HBA (ρ=0.68) "
        "ve MW-RotBonds (ρ=0.64); bu durum çoklu doğrusal bağlantı (multicollinearity) açısından "
        "dikkate alınmalıdır. Her kutunun üzerine gel, tam değeri gör."
    )
    st.plotly_chart(yap_heatmap(df), width="stretch")

# ============================================================
# TAB 2: Hipotez Testleri
# ============================================================
with tab2:
    st.success(
        "🔑 En önemli bulgu: Hangi proteine bakıldığı, pKi üzerinde tek başına herhangi bir "
        "moleküler özellikten çok daha belirleyici (Kruskal-Wallis H=12322.09, p≈0).",
        icon="🔑",
    )

    st.subheader("İstatistiksel Hipotez Testleri")
    st.markdown(
        "pKi normal dağılmadığı için (Shapiro-Wilk, p≈0) parametrik olmayan testler "
        "(Spearman korelasyonu, Kruskal-Wallis) kullanılmıştır."
    )

    with st.expander("📋 Hipotezler", expanded=True):
        st.markdown(
            "**H1:** MW arttıkça pKi artar.\n"
            "Doğrulandı — zayıf pozitif ilişki (ρ=0.204).\n\n"
            "**H2:** LogP arttıkça pKi artar.\n"
            "İlişki neredeyse yok (ρ=0.010).\n\n"
            "**H3:** TPSA arttıkça pKi azalır.\n"
            "Yön beklentisi yanlış çıktı — tam tersi zayıf pozitif ilişki bulundu (ρ=0.205).\n\n"
            "**H4:** HBA arttıkça pKi artar.\n"
            "Doğrulandı — zayıf pozitif ilişki (ρ=0.196).\n\n"
            "**H5:** Farklı proteinlerde pKi dağılımı değişir.\n"
            "Doğrulandı — proteine göre pKi belirgin şekilde farklı (Kruskal-Wallis, p≈0)."
        )

    hipotez_sonuclari = pd.DataFrame({
        "Hipotez": ["H1", "H2", "H3", "H4", "H5"],
        "Değişken": ["MW vs pKi", "LogP vs pKi", "TPSA vs pKi", "HBA vs pKi", "Protein grubu vs pKi"],
        "Korelasyon / İstatistik": ["ρ = 0.204", "ρ = 0.010", "ρ = 0.205", "ρ = 0.196", "H = 12322.09"],
        "p-değeri": ["≈0.000", "2.37e-12", "≈0.000", "≈0.000", "≈0.000"],
        "Sonuç": ["Anlamlı", "Anlamlı (etkisi ihmal edilebilir)", "Anlamlı (yön beklenenin tersi)", "Anlamlı", "Anlamlı"],
    })
    st.table(hipotez_sonuclari.set_index("Hipotez"))

    st.markdown("### Hipotezlerin Detaylı Açıklaması")

    st.markdown("**H1 — Molekül Ağırlığı (MW) vs pKi**")
    st.markdown(
        "*Hipotez:* Daha büyük (ağır) moleküllerin proteine daha güçlü bağlanması beklenir, "
        "çünkü büyük moleküller proteinle daha fazla temas noktası kurabilir.\n\n"
        "*Sonuç:* ρ=0.204, p≈0 → İstatistiksel olarak anlamlı, zayıf pozitif bir ilişki var. "
        "Hipotez yön olarak destekleniyor ama ilişki güçlü değil — MW tek başına bağlanmayı "
        "belirlemiyor, sadece kısmi bir etkisi var."
    )

    st.markdown("**H2 — LogP (Yağda Çözünürlük) vs pKi**")
    st.markdown(
        "*Hipotez:* Daha lipofilik (yağda çözünür) moleküllerin hücre zarını daha kolay "
        "geçip proteine ulaşacağı, dolayısıyla daha güçlü bağlanacağı varsayıldı.\n\n"
        "*Sonuç:* ρ=0.010, p=2.37e-12 → p-değeri anlamlı çıksa da korelasyon neredeyse sıfır. "
        "Büyük örneklem boyutu (539.218 satır) çok küçük etkileri bile 'anlamlı' gösterebiliyor; "
        "burada pratik olarak LogP'nin pKi üzerinde tek başına bir etkisi yok denebilir."
    )

    st.markdown("**H3 — TPSA vs pKi**")
    st.markdown(
        "*Hipotez:* Yüksek TPSA'nın (kutupsal yüzey alanı) hücre geçirgenliğini zorlaştırıp "
        "bağlanmayı zayıflatacağı, yani TPSA arttıkça pKi'nin azalacağı bekleniyordu.\n\n"
        "*Sonuç:* ρ=0.205, p≈0 → İlişki anlamlı ama **yönü ters** çıktı: TPSA arttıkça pKi "
        "de artıyor. Bu, başlangıç hipotezimizin desteklenmediği, verinin bizi düzelttiği "
        "bir örnek — raporda bu dürüstçe belirtilmiştir."
    )

    st.markdown("**H4 — Hidrojen Bağı Alıcı Sayısı (HBA) vs pKi**")
    st.markdown(
        "*Hipotez:* Daha fazla hidrojen bağı alıcısına sahip moleküllerin proteinle daha "
        "fazla etkileşim kurup daha güçlü bağlanacağı varsayıldı.\n\n"
        "*Sonuç:* ρ=0.196, p≈0 → Anlamlı, zayıf pozitif ilişki. Hipotez yön olarak destekleniyor."
    )

    st.markdown("**H5 — Protein Hedefi vs pKi**")
    st.markdown(
        "*Hipotez:* Farklı protein hedeflerinin yapısal farklılıkları nedeniyle, aynı tür "
        "moleküllere karşı farklı ortalama bağlanma güçleri göstereceği varsayıldı.\n\n"
        "*Sonuç:* Kruskal-Wallis H=12322.09, p≈0 → Çok güçlü şekilde anlamlı. Bu, veri "
        "setindeki en net bulgulardan biri: **hangi proteine bakıldığı**, pKi üzerinde "
        "molekülün kendi özelliklerinden bile daha belirleyici olabiliyor."
    )

    st.info(
        "H3 için TPSA arttıkça pKi'nin azalması beklenmişti, ancak sonuç zayıf da olsa "
        "pozitif çıktı (ρ=0.205) — hipotez yön bakımından desteklenmedi.",
        icon="💡",
    )
    st.warning(
        "p<0.05 olması tek başına yeterli değildir: 539.218 gözlemlik büyük örneklemde "
        "çok küçük etkiler bile anlamlı çıkabilir. Korelasyon katsayısının büyüklüğüne "
        "(etki büyüklüğü) de bakılmalıdır.",
        icon="⚠️",
    )

# ============================================================
# TAB 3: Makine Öğrenmesi
# ============================================================
with tab3:
    st.success(
        "🏆 En iyi model: **Random Forest** (Test R²=0.517) — ama varyansın hâlâ yalnızca "
        "yarısını açıklıyor, çünkü modelde protein kimliği yer almıyor.",
        icon="🏆",
    )

    st.subheader("Model Karşılaştırması")
    sonuc_tablosu = pd.DataFrame({
        "Model": ["Random Forest", "Decision Tree", "Gradient Boosting", "Linear Regression", "Ridge", "Lasso"],
        "Test R²": [0.5171, 0.3340, 0.1486, 0.0486, 0.0486, 0.0481],
        "RMSE": [1.0063, 1.1818, 1.3362, 1.4125, 1.4125, 1.4129],
        "MAE": [0.7425, 0.8261, 1.0904, 1.1594, 1.1594, 1.1605],
    }).sort_values("Test R²", ascending=False)
    st.table(sonuc_tablosu.set_index("Model"))
    st.caption("En iyi model **Random Forest** (Test R²=0.517), ama varyansın hâlâ yalnızca yarısını açıklıyor.")

    st.plotly_chart(yap_r2_bar(sonuc_tablosu), width="stretch")

    st.markdown("#### Eğitim vs Test Performansı — Aşırı/Yetersiz Öğrenme")
    egitim_test_tablosu = pd.DataFrame({
        "Model": ["Random Forest", "Decision Tree", "Gradient Boosting"],
        "Eğitim R²": [0.800, 0.829, 0.151],
        "Test R²": [0.517, 0.334, 0.149],
    })
    st.table(egitim_test_tablosu.set_index("Model"))
    st.markdown(
        "**Random Forest**'ta eğitim R² (0.800), test R²'den (0.517) belirgin şekilde yüksek — "
        "bir miktar aşırı öğrenme (overfitting) var, ama model yine de en iyi genelleyen model. "
        "**Decision Tree**'de bu fark çok daha büyük (eğitim 0.829, test 0.334): model eğitim "
        "verisini neredeyse ezberlemiş, yeni veriye iyi genelleyememiş. **Gradient Boosting**'de "
        "ise tam tersi bir durum var — hem eğitim (0.151) hem test (0.149) performansı düşük, "
        "bu da modelin yetersiz öğrendiğine (underfitting) işaret ediyor. Linear Regression, "
        "Ridge ve Lasso ise birbirine çok yakın ve düşük sonuçlar veriyor; bu da veride güçlü "
        "doğrusal bir örüntü olmadığını doğruluyor."
    )

    st.subheader("Random Forest — Özellik Önem Dereceleri")
    st.caption("Modelin tahmin yaparken en çok güvendiği değişkenler: MW, TPSA ve LogP. Üzerine gelince tam puanı gör.")

    with st.spinner("Model yükleniyor..."):
        rf_model = model_yukle()

    onem = pd.DataFrame({
        "Özellik": features, "Önem": rf_model.feature_importances_
    }).sort_values("Önem", ascending=False)
    st.plotly_chart(
        yap_feature_importance(tuple(onem["Özellik"]), tuple(onem["Önem"])),
        width="stretch",
    )

# ============================================================
# TAB 4: Canlı Tahmin Aracı
# ============================================================
with tab4:
    st.subheader("Kendi Molekül Özelliklerini Gir, pKi Tahmini Al")
    st.caption("Random Forest modeli kullanılarak canlı tahmin yapılır.")

    rf_model = model_yukle()

    c1, c2, c3 = st.columns(3)
    mw = c1.slider("MW (Moleküler Ağırlık)", 100.0, 1000.0, 450.0)
    logp = c1.slider("LogP", -5.0, 10.0, 3.0)
    tpsa = c2.slider("TPSA", 0.0, 300.0, 90.0)
    hbd = c2.slider("HBD", 0, 10, 2)
    hba = c3.slider("HBA", 0, 15, 5)
    rotbonds = c3.slider("RotBonds", 0, 20, 5)

    girdi = pd.DataFrame([[mw, logp, tpsa, hbd, hba, rotbonds]], columns=features)
    tahmin = rf_model.predict(girdi)[0]

    st.metric("Tahmini pKi", f"{tahmin:.2f}")

    if tahmin >= 8:
        st.success("Güçlü bağlanma tahmini 🎯")
        st.balloons()
    elif tahmin >= 6:
        st.info("Orta düzey bağlanma tahmini")
    else:
        st.warning("Zayıf bağlanma tahmini")

    st.caption(
        "Not: Model açıklama gücü sınırlı (Test R²=0.52), çünkü protein kimliği "
        "özellik olarak dahil edilmemiştir — bu tahmini yaklaşık bir gösterge olarak değerlendir."
    )

# ============================================================
# TAB 5: Sonuç
# ============================================================
with tab5:
    st.subheader("📌 Genel Sonuç")

    st.markdown(
        "Bu çalışmada, kinaz ailesi proteinlere ait **539.218 molekül-protein çiftinden** "
        "(**1.573 benzersiz protein**) oluşan bir veri seti, üç farklı halka açık kaynaktan "
        "(BindingDB, Davis, KIBA) derlenerek oluşturulmuş; veriler temizlenmiş, ortak bir pKi "
        "ölçeğine getirilmiş ve RDKit ile altı fizikokimyasal tanımlayıcı hesaplanmıştır. "
        "Ortalama pKi **6.77** olarak bulunmuştur."
    )

    st.markdown(
        "Keşifsel analiz ve hipotez testleri, incelenen fizikokimyasal özelliklerin "
        "(MW, TPSA, HBA) pKi ile istatistiksel olarak anlamlı ama pratikte zayıf ilişkiler "
        "taşıdığını; buna karşılık **protein hedefinin pKi üzerinde çok daha güçlü bir etkiye "
        "sahip olduğunu** (Kruskal-Wallis H=12322.09) göstermiştir. Bu bulgu, makine öğrenmesi "
        "aşamasında modellerin neden sınırlı kaldığını açıklamaktadır."
    )

    st.markdown(
        "Altı makine öğrenmesi modeli karşılaştırıldığında **Random Forest** en iyi performansı "
        "göstermiş (**Test R²=0.517**), ancak bu değer varyansın yalnızca yarısını açıklayabildiğini "
        "ortaya koymaktadır. Bu sınırlılığın en olası nedeni, modele yalnızca molekülün kendi "
        "özelliklerinin verilmesi, protein kimliğinin hiç dahil edilmemesidir."
    )

    st.success(
        "Sonuç olarak: fizikokimyasal tanımlayıcılar bağlanma afinitesi tahmininde kısmi bir "
        "gösterge olabilir, ancak güvenilir bir tahmin için protein bilgisinin de modele "
        "katılması gerekmektedir.",
        icon="✅",
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Nihai Veri Seti", "539.218 satır")
    col_b.metric("En İyi Model", "Random Forest")
    col_c.metric("Test R²", "0.517")

    st.divider()
    st.caption(
        "Kod, temizlenmiş veri seti ve bu dashboard GitHub'da mevcuttur: "
        "github.com/cekereklibeyza/protein-ilac-baglanma-afinitesi-tahmini"
    )
