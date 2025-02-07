import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io

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
# CSVテンプレートダウンロードボタンの設置
st.title("回帰分析 WEB アプリ")
st.markdown("### CSVテンプレートのダウンロード")
st.write("""
以下のボタンをクリックすると、各項目の説明が記載されたCSVファイルのひな形をダウンロードできます。  
このテンプレートの**最初の3行**には各項目の説明が書かれており、**4行目以降**に実際のデータ（目的変数 Y と説明変数 X）が記載されます。  
ご自身のデータを入力する際の参考にしてください。
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
※ CSVファイルの最初の8行は説明文として扱われるため、**ヘッダー行は9行目**に記載してください。
""")

# ------------------------------------------
# CSVファイルアップロードとデータ読み込み
st.sidebar.header("1. データのアップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード（ヘッダーは4行目）", type=["csv"])

if uploaded_file is not None:
    try:
        # ヘッダーが9行目にあるので、skiprows=3として読み込む（エンコーディングはutf-8-sig）
        df = pd.read_csv(uploaded_file, skiprows=8, encoding='utf-8-sig')
        st.write("### アップロードされたデータ（一部）")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        st.stop()

    # ------------------------------------------
    # 変数の選択（サイドバー）
    st.sidebar.header("2. 変数の選択")
    all_columns = df.columns.tolist()

    target_var = st.sidebar.selectbox("目的変数（Y）を選択", all_columns)
    feature_vars = st.sidebar.multiselect(
        "説明変数（X）を選択（複数選択可）",
        [col for col in all_columns if col != target_var]
    )

    if not feature_vars:
        st.warning("説明変数（X）を少なくとも1つ選択してください。")
    else:
        if st.sidebar.button("回帰分析を実行"):
            # ------------------------------------------
            # データ抽出と前処理
            X = df[feature_vars]
            y = df[target_var]

            # 欠損値の処理（平均値補完）
            if X.isnull().any().any() or y.isnull().any():
                st.warning("欠損値が検出されたため、平均値で補完します。")
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())

            # ------------------------------------------
            # 線形回帰モデルの構築
            model = LinearRegression()
            model.fit(X, y)

            # 予測値の算出
            y_pred = model.predict(X)

            # 評価指標の算出
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            st.subheader("モデル評価")
            st.write(f"平均二乗誤差 (MSE): **{mse:.4f}**")
            st.write(f"決定係数 (R²): **{r2:.4f}**")

            # ------------------------------------------
            # 統計初心者向けの全体の解説を返す関数
            def explain_relationship(r2_value):
                if r2_value >= 0.7:
                    explanation = ("この結果は、説明変数と目的変数の間に**かなり強い関係**があることを示しています。")
                elif r2_value >= 0.5:
                    explanation = ("この結果は、説明変数と目的変数の間に**おそらく関係**があることを示唆しています。\n"
                                   "また、関係性がある可能性は高いですが、他の要因も影響している可能性があります。")
                elif r2_value >= 0.3:
                    explanation = ("この結果は、説明変数と目的変数の間に**関係がある可能性もある**ことを示していますが、"
                                   "その関係はそこまで明確ではなく、予測精度は低めです。")
                elif r2_value >= 0.1:
                    explanation = ("この結果は、説明変数と目的変数の間に**あまり関係がない**ことを示しています。\n"
                                   "関係は限定的で、他の要因が大きく影響している可能性があります。")
                else:
                    explanation = ("この結果は、説明変数と目的変数の間に**全く関係がない**、または非常に弱い関係しかないことを示しています。")
                return explanation

            # 統計解説の表示（全体の結果）
            explanation_text = explain_relationship(r2)
            st.markdown("**【統計解説】**")
            st.write(explanation_text)
            # 説明変数と目的変数の欠損値を事前に補完
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            # ------------------------------------------
            # 複数の説明変数がある場合、各変数ごとの目的変数との相関関係と解説を表示する
                               
            if len(feature_vars) > 1:
            st.subheader("各説明変数と目的変数の個別の関係")

            def explain_individual_relationship(corr_value):
                abs_corr = abs(corr_value)
            if abs_corr >= 0.7:
                strength = "かなり強い関係"
            elif abs_corr >= 0.5:
                strength = "おそらく関係がある"
            elif abs_corr >= 0.3:
                strength = "関係がある可能性もある"
            elif abs_corr >= 0.1:
                strength = "あまり関係がない"
            else:
                strength = "全く関係がない"
            if corr_value > 0:
                direction = "正の相関"
            elif corr_value < 0:
                direction = "負の相関"
            else:
                direction = "相関なし"
            return f"{direction}、{strength}"

        individual_explanations = []
        for var in feature_vars:
            # 欠損値補完後のデータを用いて相関を計算
            corr_val = y.corr(X[var])
            exp_text = explain_individual_relationship(corr_val)
            individual_explanations.append({
                "変数": var,
                "相関係数": round(corr_val, 4),
                "解説": exp_text
            })
        exp_df = pd.DataFrame(individual_explanations)
        st.dataframe(exp_df)

            # ------------------------------------------
            # 回帰係数の表示
            st.subheader("回帰係数")
            coef_df = pd.DataFrame({
                "変数": feature_vars,
                "係数": model.coef_
            })
            st.dataframe(coef_df)
            st.write(f"切片: **{model.intercept_:.4f}**")

            # ------------------------------------------
            # 予測結果の可視化：実測値 vs 予測値
            st.subheader("予測結果の可視化")
            fig, ax = plt.subplots()
            ax.scatter(y, y_pred, alpha=0.7, edgecolors="b")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax.set_xlabel("実測値")
            ax.set_ylabel("予測値")
            ax.set_title("実測値と予測値の比較")
            st.pyplot(fig)
            st.write("""
**図の見方：**
- **横軸：** 実際に観測された値（実測値）
- **縦軸：** モデルが予測した値（予測値）
- **赤い点線：** 理想的な一致（実測値＝予測値）のライン
  
点が赤い点線に近いほど、モデルの予測が正確であることを意味します。
""")

            # 説明変数が1つの場合の散布図と回帰直線の表示
            if len(feature_vars) == 1:
                st.subheader(f"{feature_vars[0]} と {target_var} の関係")
                fig2, ax2 = plt.subplots()
                sns.regplot(x=feature_vars[0], y=target_var, data=df, ax=ax2, line_kws={"color": "red"})
                ax2.set_title("説明変数と目的変数の散布図と回帰直線")
                st.pyplot(fig2)
                st.write("""
**図の見方：**
- **横軸：** 説明変数の値
- **縦軸：** 目的変数の値
- **赤い直線：** データの傾向（回帰直線）
  
点が直線に沿って分布していれば、説明変数と目的変数との関係が強いと考えられます。
""")
            else:
                # 複数の説明変数がある場合、説明変数間の相関ヒートマップを表示
                st.subheader("説明変数間の相関関係")
                corr = df[feature_vars].corr()
                fig3, ax3 = plt.subplots()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
                ax3.set_title("各説明変数間の相関ヒートマップ")
                st.pyplot(fig3)
                st.write("""
**図の見方：**
- 数字は各変数間のPearsonの相関係数を示しています。
- **1に近い：** 非常に強い正の相関（両方の値が共に増加する傾向）
- **-1に近い：** 非常に強い負の相関（片方の値が増加すると、もう片方が減少する傾向）
- **0に近い：** 相関がほとんどない
  
色が濃いほど、変数間の関係が強いことを意味します。
""")
