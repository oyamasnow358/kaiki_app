import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os  # osモジュールのインポートを追加
import matplotlib.font_manager as fm
import matplotlib as mpl

# フォント設定
font_path = os.path.abspath("ipaexg.ttf")  # 絶対パス
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams["font.family"] = font_prop.get_name()
    plt.rc("font", family=font_prop.get_name())  # 追加
    st.write(f"✅ フォント設定: {mpl.rcParams['font.family']}")
else:
    st.error("❌ フォントファイルが見つかりません。")

# 相関関係の解釈関数
def explain_relationship(corr_value):
    if abs(corr_value) >= 0.7:
        return "かなり強い関係がある"
    elif abs(corr_value) >= 0.5:
        return "おそらく関係がある"
    elif abs(corr_value) >= 0.3:
        return "関係がある可能性がある"
    elif abs(corr_value) >= 0.1:
        return "あまり関係がない"
    else:
        return "ほとんど関係がない"
# ------------------------------------------
# CSVテンプレート作成用の文字列
template_csv = """このCSVファイルは、回帰分析用のデータひな形です。
'行10行以降に実際のデータを入力してください。
'【各列の説明】
'
'Y: 目的変数（例: 売上、テストの得点など）
'X1, X2, ...: 説明変数（例: 気温、広告費、出席率など）
※ 例として、以下はサンプルデータです。

目的変数,説明変数1,説明変数2,説明変数3,説明変数4,説明変数5
100,10,5,3,2,4
110,12,6,4,2,3
105,11,5,2,2,3
"""
st.title("回帰分析 WEB アプリ")

# 初心者向け説明の表示切り替え
if "show_explanation" not in st.session_state:
           st.session_state.show_explanation = False
        # ボタンを押すたびにセッションステートを切り替える
if st.button("説明を表示/非表示"):
           st.session_state.show_explanation = not st.session_state.show_explanation

         # セッションステートに基づいて説明を表示
if st.session_state.show_explanation:
           st.markdown("""
          ## **回帰分析とは？**  
                       
           - ### **1. そもそも「回帰分析」って何？**

            回帰分析とは、「あるデータ（結果）が、ほかのデータ（要因）によってどのように変わるか」を調べる方法です。
            例えば、こんなことを知りたいときに使えます：

            ・気温が変わるとアイスの売り上げはどう変わるか？  
            ・勉強時間が長いほどテストの点数は上がるのか？  

            このように「1つの原因（説明変数）」が「1つの結果（目的変数）」に影響を与えるかを見るのが単回帰分析です。  
                       
           - ### **2. じゃあ「多変量回帰分析」って？**

            多変量回帰分析は、「複数の要因」が「1つの結果」にどれくらい影響を与えているかを調べる方法です。

            例えば、テストの点数に影響を与える要因は1つだけではありませんよね？

            ・勉強時間  
            ・睡眠時間  
            ・スマホの使用時間  
            ・塾に通っているか など  

            こうした**複数の要因（説明変数）**が、**テストの点数（目的変数）**にどのくらい関係しているのかを数式で表すのが「多変量回帰分析」です。  
                       
           - ### **3. 数式で表すと？**

            多変量回帰分析の基本的な式はこうなります：  
                                y=a1x1+a2x2+a3x3+⋯+b  
            
            ・yy：結果（目的変数）→ 例えば「テストの点数」  
            ・x1,x2,x3x1​,x2​,x3​ …：要因（説明変数）→ 例えば「勉強時間」「睡眠時間」「スマホ使用時間」など  
            ・a1,a2,a3a1​,a2​,a3​ …：それぞれの要因が結果に与える影響の大きさ（重み）  
            ・bb：一定の値（定数）

            例えば、テストの点数を予測する場合、こんな式になるかもしれません：  
                        テストの点数=5×勉強時間+2×睡眠時間−3×スマホ時間+50  
           

             この式が意味するのは：

             ・勉強時間が1時間増えると、点数は5点上がる  
             ・睡眠時間が1時間増えると、点数は2点上がる  
             ・スマホ時間が1時間増えると、点数は3点下がる  
             ・何もしなくても最低50点は取れる（かもしれない）    

           -  ### **4. どうやって役に立つの？**

             多変量回帰分析を使うと、いろんなことがわかります！  
             ✔ どの要因が一番影響が大きいのか？  
             ✔ どの要因を変えれば結果がよくなるのか？  
             ✔ 未来の結果を予測できる！  

             例えば、ある生徒の勉強時間、睡眠時間、スマホ時間がわかれば、その生徒のテストの点数を予測することもできます！""")  
           
# ------------------------------------------
# CSVテンプレートダウンロードボタンの設置

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
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, skiprows=8, encoding='utf-8-sig')
        st.write("### アップロードされたデータ")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        st.stop()

    # --------------------------------------
    # 変数選択
    st.sidebar.header("2. 変数の選択")
    all_columns = df.columns.tolist()

    if len(all_columns) < 2:
        st.error("データに2列以上の変数が必要です。")
        st.stop()

    # "目的変数" という列があればそれを選択、なければ最初の列をデフォルトにする
    default_target_var = "目的変数" if "目的変数" in all_columns else all_columns[0]

    # 目的変数の選択
    target_var = st.sidebar.selectbox("目的変数（Y）を選択", all_columns, index=all_columns.index(default_target_var))

    # 説明変数（X）の選択肢は、目的変数（Y）を除外
    feature_vars = st.sidebar.multiselect(
        "説明変数（X）を選択（複数選択可）",
        [col for col in all_columns if col != target_var]
    )

    if not feature_vars:
        st.warning("説明変数を選択してください。")

    else:
        if st.sidebar.button("回帰分析を実行"):
            # ----------------------------------
            # データ抽出と前処理
            X = df[feature_vars]
            y = df[target_var]
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            # ----------------------------------
            # 線形回帰モデル
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # モデル評価
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            st.subheader("モデル評価")
            st.write(f"平均二乗誤差 (MSE): **{mse:.4f}**")
            st.write(f"決定係数 (R²): **{r2:.4f}**")

            # ----------------------------------
            
            # 回帰係数と目的変数との相関
            st.subheader("回帰係数と目的変数との相関")
            coef_df = pd.DataFrame({
            "変数": feature_vars,
            "回帰係数": model.coef_.astype(float).round(4),
            "目的変数との相関係数": [df[[col, target_var]].corr().iloc[0, 1].round(4) for col in feature_vars]
            })
            coef_df["相関の解釈"] = coef_df["目的変数との相関係数"].apply(explain_relationship)

            st.dataframe(coef_df)
            st.write(f"切片: **{model.intercept_:.4f}**")


            # ------------------------------------------
            # 予測結果の可視化：実測値 vs 予測値
            st.subheader("予測結果可視化")
            fig, ax = plt.subplots()
            ax.scatter(y, y_pred, alpha=0.7, edgecolors="b")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax.set_xlabel("実測値", fontproperties=font_prop)  # ← 日本語フォント適用
            ax.set_ylabel("予測値", fontproperties=font_prop)
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
                st.pyplot(fig2)
                st.write("""
**図の見方：**
- **横軸：** 説明変数の値
- **縦軸：** 目的変数の値
- **赤い直線：** データの傾向（回帰直線）
  
点が直線に沿って分布していれば、説明変数と目的変数との関係が強いと考えられます。
""")
            else:
                st.subheader("説明変数間の相関")
    
    # 相関行列の作成
                corr = df[feature_vars].corr()
    
    # カラム名を日本語化（説明変数名をリスト化）
                corr.index = [f"説明変数{i+1}" for i in range(len(corr.index))]
                corr.columns = [f"説明変数{i+1}" for i in range(len(corr.columns))]
               
    # 相関行列をヒートマップとして描画
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3, fmt=".2f", linewidths=0.5,ax=ax3)
                
                   # 軸ラベルの日本語設定
                ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontproperties=font_prop)
                ax3.set_yticklabels(ax3.get_yticklabels(), fontproperties=font_prop)
                
                # 日本語フォントを適用失敗
                #ax3.set_xlabel("説明変数", fontproperties=font_prop)
                #ax3.set_ylabel("説明変数", fontproperties=font_prop)

                st.pyplot(fig3)
                st.write("""
**図の見方：**
- 数字は各変数間のPearsonの相関係数を示しています。
- **1に近い：** 非常に強い正の相関（両方の値が共に増加する傾向）
- **-1に近い：** 非常に強い負の相関（片方の値が増加すると、もう片方が減少する傾向）
- **0に近い：** 相関がほとんどない
  
色が濃いほど、変数間の関係が強いことを意味します。
""")
