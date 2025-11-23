import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import io

# --- 1. 初期設定 ---
st.set_page_config(
    page_title="因果・相関分析マスター",
    page_icon="🔍",
    layout="wide"
)

# --- 2. 計算ロジック ---

def calculate_partial_correlation(df, x, y, covar):
    try:
        temp_df = df[[x, y, covar]].dropna()
        if len(temp_df) < 3: return np.nan, np.nan

        r_xy = temp_df[x].corr(temp_df[y])
        r_xz = temp_df[x].corr(temp_df[covar])
        r_yz = temp_df[y].corr(temp_df[covar])
        
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        if denominator == 0: return np.nan, np.nan
        return numerator / denominator, r_xy
    except:
        return np.nan, np.nan

def create_csv_template():
    template_df = pd.DataFrame({
        '国語テスト(点)': [80, 65, 92, 75, 58, 85, 70, 95, 60, 78],
        '読書量(冊)': [5, 2, 8, 4, 1, 6, 3, 10, 1, 5],
        '語彙力スコア': [60, 45, 70, 55, 40, 62, 50, 75, 38, 58],
        'スマホ時間(分)': [60, 120, 30, 90, 150, 50, 100, 20, 160, 80]
    })
    return template_df.to_csv(index=False)

# --- 3. メイン処理 ---

def main():
    st.title("🔍 因果・相関分析マスター")
    st.markdown("""
    データの「関係性」には種類があります。目的に合わせてタブを切り替えてください。
    """)
    
    # --- サイドバー: ナビゲーションガイド ---
    with st.sidebar:
        st.header("🧭 迷ったらココを読む")
        st.info("""
        **Q. どっちを信じればいい？**
        
        👉 **「成績を上げたい」なら...**
        **【STEP 2: 犯人探し】** を信じてください。見せかけの要因をいくら改善しても結果は変わりません。
        
        👉 **「来月の結果を知りたい」なら...**
        **【STEP 3: 未来予測】** を信じてください。原因が何であれ、データ上の傾向を使えば予測は当たります。
        """)
        
        st.divider()
        st.header("📂 データ入力")
        uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
        
        st.markdown("##### テスト用データ")
        csv_text = create_csv_template()
        st.download_button("📥 サンプルCSV", csv_text.encode('utf-8-sig'), "sample_data.csv", "text/csv")

    # データ読み込み
    if uploaded_file:
        try: df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except: 
            try: df = pd.read_csv(uploaded_file, encoding='shift-jis')
            except: st.error("読込エラー"); return
    else:
        df = pd.read_csv(io.StringIO(create_csv_template()))
        st.info("💡 現在はサンプルデータモードです。")

    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.shape[1] < 2:
        st.warning("⚠️ 数値列が2つ以上必要です。")
        return

    # --- タブ名の変更：目的別に ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 STEP 1: 現状を見る (相関)", 
        "🕵️ STEP 2: 犯人を探す (因果)", 
        "🔮 STEP 3: 未来を読む (予測)", 
        "📋 データ一覧"
    ])

    # ==========================================
    # Tab 1: 相関 (現状把握)
    # ==========================================
    with tab1:
        st.subheader("📊 データの「つながり」を確認する")
        st.markdown("ここでは単純に**「Aが多いとき、Bも多いか？」**だけを見ます。理由（因果）は考えません。")
        
        corr_matrix = df_numeric.corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto", 
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("赤＝一緒に増える関係、青＝逆の動きをする関係")

    # ==========================================
    # Tab 2: 因果 (犯人探し) - 最重要
    # ==========================================
    with tab2:
        st.subheader("🕵️ 結果を変えるための「本当の原因」を探す")
        st.markdown("""
        **「指導や対策」を考えるならココ！**
        一見関係ありそうでも、別の要因（黒幕）がいる場合、対策しても無駄になります。
        """)

        c1, c2, c3 = st.columns(3)
        if len(df_numeric.columns) >= 3:
            with c1: tx = st.selectbox("対策したい要因 (X)", df_numeric.columns, 0)
            with c2: ty = st.selectbox("良くしたい結果 (Y)", df_numeric.columns, 1)
            with c3: 
                cands = [c for c in df_numeric.columns if c not in [tx, ty]]
                tz = st.selectbox("疑わしい黒幕 (Z)", cands) if cands else None

            st.divider()

            if tx and ty and tz:
                if tx == ty:
                    st.warning("要因と結果は別の変数にしてください")
                else:
                    p_corr, raw_corr = calculate_partial_correlation(df_numeric, tx, ty, tz)
                    
                    if np.isnan(p_corr):
                        st.error("計算できませんでした")
                    else:
                        # 結果表示
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.metric("表面上の関係 (相関)", f"{raw_corr:.3f}")
                        with col_res2:
                            st.metric(f"黒幕({tz})を除いた本当の関係", f"{p_corr:.3f}", 
                                      delta=f"{p_corr - raw_corr:.3f}", delta_color="inverse")
                        
                        # 親しみやすい診断メッセージ
                        diff = abs(raw_corr - p_corr)
                        st.markdown("### 📝 分析結果")
                        
                        if diff > 0.3 and abs(p_corr) < 0.2:
                            st.error(f"""
                            **⚠️ これは「見せかけ」です！ (疑似相関)**
                            
                            「{tx}」と「{ty}」に関係があるように見えますが、実は両方とも「{tz}」の影響を受けているだけです。
                            **【結論】 「{tx}」を頑張って改善しても、「{ty}」はほとんど上がらないでしょう。**
                            対策するなら「{tz}」の方にアプローチすべきです。
                            """)
                        elif diff < 0.1:
                            st.success(f"""
                            **✅ これは「本物」の可能性が高いです！**
                            
                            「{tz}」の影響を考慮しても、関係性は消えませんでした。
                            **【結論】 「{tx}」を改善すれば、「{ty}」も良くなる可能性が高いです。**
                            自信を持って指導に取り入れてください。
                            """)
                        else:
                            st.warning(f"""
                            **🤔 一部影響しています**
                            
                            「{tz}」も関係していますが、「{tx}」自身の効果もありそうです。
                            """)
        else:
            st.warning("変数が3つ以上必要です")

    # ==========================================
    # Tab 3: 予測 (回帰)
    # ==========================================
    with tab3:
        st.subheader("🔮 データの傾向から「未来」を予測する")
        st.markdown("""
        **「見込み」を知りたいならココ！**
        因果関係がどうあれ、「今のデータ傾向だと、結果はどうなるか？」を正確に計算します。
        """)
        
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1: x_col = st.selectbox("入力データ (X)", df_numeric.columns, 0, key='reg_x')
        with c_sel2: y_col = st.selectbox("予測したいもの (Y)", df_numeric.columns, 1, key='reg_y')

        if x_col == y_col:
            st.warning("XとYは別の変数を選んでください。")
        else:
            plot_df = df.dropna(subset=[x_col, y_col])
            if len(plot_df) > 0:
                X = sm.add_constant(plot_df[x_col])
                model = sm.OLS(plot_df[y_col], X).fit()
                
                slope = model.params.iloc[1]
                intercept = model.params.iloc[0]
                r2 = model.rsquared

                # グラフ
                fig = px.scatter(
                    plot_df, x=x_col, y=y_col, trendline="ols",
                    trendline_color_override="red", hover_data=df.columns
                )
                fig.update_layout(title=f"予測モデル: {x_col} → {y_col}")
                st.plotly_chart(fig, use_container_width=True)

                # レポート
                st.markdown("### 📝 予測レポート")
                col_rep1, col_rep2 = st.columns(2)
                
                with col_rep1:
                    st.metric("予測の正確さ (決定係数)", f"{r2*100:.1f}%")
                    if r2 > 0.5:
                        st.success("かなり正確に予測できます。")
                    else:
                        st.warning("予測のズレが大きいです。")
                        
                with col_rep2:
                    st.info(f"💡 **注意点**: \nここで「正確に予測できる」と出ても、STEP 2で「見せかけ」と判定された場合は、**{x_col}を無理やり増やしても結果は変わりません。**")

                # シミュレーター
                st.markdown("---")
                st.write(f"**👇 スライダーでシミュレーション ({x_col}を変えるとどうなる？)**")
                
                user_x = st.slider(
                    f"{x_col} の値",
                    float(plot_df[x_col].min()),
                    float(plot_df[x_col].max()),
                    float(plot_df[x_col].mean())
                )
                pred_y = slope * user_x + intercept
                
                st.metric(f"予測される {y_col}", f"{pred_y:.1f}")

    # ==========================================
    # Tab 4: データ
    # ==========================================
    with tab4:
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()