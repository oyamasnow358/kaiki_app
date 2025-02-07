import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# CSVテンプレート作成用の文字列
template_csv = """このCSVファイルは、回帰分析用のデータひな形です。
'行10行以降に実際のデータを入力してください。
'【各列の説明】
'ID: レコード番号（任意）
'Y: 目的変数（例: 売上、テストの得点など）
'X1, X2, ...: 説明変数（例: 気温、広告費、出席率など）
※ 例として、以下はサンプルデータです。

ID,目的変数,説明変数1,説明変数2,説明変数3,説明変数4,説明変数5
1,100,10,5,3,2,4
2,110,12,6,4,2,3
3,105,11,5,2,2,3
"""

# ------------------------------------------
# アプリケーションヘッダー
st.title("回帰分析 WEB アプリ")
st.markdown("### CSVテンプレートのダウンロード")
st.write("""
以下のボタンをクリックすると、各項目の説明が記載されたCSVファイルのひな形をダウンロードできます。  
このテンプレートの**最初の3行**には各項目の説明が書かれており、**4行目以降**に実際のデータ（目的変数 Y と説明変数 X）が記載されます。
""")
st.download_button(
    label="CSVテンプレートをダウンロード",
    data=template_csv.encode('utf-8-sig'),
    file_name="template.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("### 回帰分析の実行方法")
st.write("""
1. サイドバーから、上記テンプレートに沿った形式のCSVファイルをアップロードしてください。  
2. アップロード後、**目的変数（Y）**と**説明変数（X）**を選択します。  
3. 「回帰分析を実行」ボタンを押すと、モデルの係数や評価指標、可視化グラフが表示されます。
""")

# ------------------------------------------
# CSVファイルアップロード
st.sidebar.header("1. データのアップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード（ヘッダーは9行目）", type=["csv"])

if not uploaded_file:
    st.warning("CSVファイルをアップロードしてください。")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, skiprows=8, encoding='utf-8-sig')
    st.write("### アップロードされたデータ（一部）")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"ファイル読み込みエラー: {e}")
    st.stop()

# ------------------------------------------
# 変数の選択
st.sidebar.header("2. 変数の選択")
all_columns = df.columns.tolist()

target_var = st.sidebar.selectbox("目的変数（Y）を選択", all_columns)
feature_vars = st.sidebar.multiselect(
    "説明変数（X）を選択（複数選択可）",
    [col for col in all_columns if col != target_var]
)

if not target_var:
    st.warning("目的変数を選択してください。")
    st.stop()

if not feature_vars:
    st.warning("説明変数を選択してください。")
    st.stop()

# ------------------------------------------
# 回帰分析の実行
if st.sidebar.button("回帰分析を実行"):
    X = df[feature_vars]
    y = df[target_var]

    # 欠損値処理
    if X.isnull().any().any() or y.isnull().any():
        st.warning("欠損値が検出されたため、平均値で補完します。")
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

    # モデル構築
    model = LinearRegression()
    model.fit(X, y)

    # トレーニング確認
    if not hasattr(model, "coef_"):
        st.error("回帰モデルがトレーニングされていません。データや選択変数を確認してください。")
        st.stop()

    # 予測と評価指標
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.subheader("モデル評価")
    st.write(f"平均二乗誤差 (MSE): **{mse:.4f}**")
    st.write(f"決定係数 (R²): **{r2:.4f}**")

    # ------------------------------------------
    # 可視化：実測値 vs 予測値
    st.subheader("予測結果の可視化")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("""
        **図の見方：**
        - **横軸：** 実測値  
        - **縦軸：** 予測値  
        - **赤い点線：** 理想的な一致ライン  
        点がこのラインに近いほど、予測が正確です。
        """)

    with col2:
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, alpha=0.7, edgecolors="b")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_xlabel("実測値")
        ax.set_ylabel("予測値")
        ax.set_title("実測値と予測値の比較")
        st.pyplot(fig)

    # ------------------------------------------
    # 説明変数間の相関ヒートマップ
    if len(feature_vars) > 1:
        st.subheader("説明変数間の相関関係")
        st.write("以下のヒートマップは、説明変数間の相関関係を視覚的に示したものです。")
        corr = df[feature_vars].corr()
        fig2, ax2 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
        ax2.set_title("各説明変数間の相関ヒートマップ")
        st.pyplot(fig2)

    # ------------------------------------------
    # 回帰係数の表示
    st.subheader("回帰係数")
    coef_df = pd.DataFrame({
        "変数": feature_vars,
        "係数": model.coef_
    })
    st.dataframe(coef_df)
    st.write(f"切片: **{model.intercept_:.4f}**")
