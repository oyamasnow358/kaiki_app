import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import io

# --- 1. 初期設定 ---
st.set_page_config(
    page_title="重回帰・要因分析マスター",
    page_icon="📊",
    layout="wide"
)

# --- 2. 関数定義 ---

def create_csv_template():
    """テンプレートCSVの生成"""
    template_df = pd.DataFrame({
        '店舗の売上(万)': [1200, 1150, 1400, 1600, 900, 1800, 1300, 1100, 1750, 1050],
        '駅からの距離(分)': [5, 7, 3, 2, 10, 1, 6, 8, 2, 9],
        '広告費用(万)': [30, 25, 40, 50, 10, 60, 35, 20, 55, 15],
        '従業員数(人)': [4, 4, 5, 6, 3, 7, 5, 3, 6, 3],
        '品揃え数(種)': [50, 45, 60, 70, 30, 80, 55, 40, 75, 35]
    })
    return template_df.to_csv(index=False)

def run_regression_analysis(df, target_col, feature_cols):
    """
    Statsmodelsを用いて重回帰分析を行い、詳細な結果を返す
    """
    try:
        # データの準備（欠損値除去）
        data = df[[target_col] + feature_cols].dropna()
        if len(data) < len(feature_cols) + 2:
            return {"status": "error", "message": "データ数が少なすぎます。変数の数より多くのデータ行が必要です。"}

        X = data[feature_cols]
        y = data[target_col]

        # 定数項（切片）の追加
        X_with_const = sm.add_constant(X)

        # 1. 通常の回帰分析（予測用）
        model = sm.OLS(y, X_with_const).fit()

        # 2. 標準化回帰係数の計算（影響度比較用）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
        
        # statsmodelsで標準化データを計算するためにDataFrame化
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        X_scaled_df = sm.add_constant(X_scaled_df) 
        model_scaled = sm.OLS(y_scaled, X_scaled_df).fit()

        # 結果の整理
        result_df = pd.DataFrame({
            "変数名": feature_cols,
            "係数 (傾き)": model.params[feature_cols],
            "標準化係数 (影響度)": model_scaled.params[feature_cols],
            "P値 (信頼度)": model.pvalues[feature_cols]
        })

        # 評価指標
        r2 = model.rsquared
        adj_r2 = model.rsquared_adj
        
        return {
            "status": "success",
            "model": model,
            "result_df": result_df,
            "r2": r2,
            "adj_r2": adj_r2,
            "data": data,
            "target": target_col,
            "features": feature_cols
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 3. メインアプリ ---

def main():
    st.title("🚀 重回帰・要因分析マスター")
    st.markdown("""
    **「結果（売上や点数）」**に対して、**「どの要因（広告や勉強時間）」**がどれくらい効いているのか？
    数式を使ってズバリ分析し、AIレポートで解説します。
    """)
    
    # --- サイドバー ---
    with st.sidebar:
        st.header("📂 データ設定")
        
        # 1. データアップロード
        uploaded_file = st.file_uploader("CSVをアップロード", type=["csv"])
        
        st.markdown("---")
        # テンプレート
        st.markdown("##### 📌 テスト用データ")
        csv_text = create_csv_template()
        st.download_button("📥 サンプルCSV", csv_text.encode('utf-8-sig'), "sample_regression.csv", "text/csv")

    # データの読み込み処理
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except:
            try: df = pd.read_csv(uploaded_file, encoding='shift-jis')
            except: st.error("読込エラー: 文字コードを確認してください"); return
    else:
        # デモモード
        df = pd.read_csv(io.StringIO(create_csv_template()))
        st.info("💡 現在はサンプルデータモードです。左側から自分のデータをアップロードできます。")

    # 数値列の抽出
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.shape[1] < 2:
        st.error("分析には数値の列が2つ以上必要です。")
        return

    # --- 変数選択エリア ---
    st.markdown("### 1. 何を分析しますか？")
    col_var1, col_var2 = st.columns(2)
    
    with col_var1:
        target_var = st.selectbox("🎯 予測したい結果 (目的変数 Y)", df_numeric.columns, index=0)
    
    with col_var2:
        feature_candidates = [c for c in df_numeric.columns if c != target_var]
        feature_vars = st.multiselect(
            "⚡ 要因と思われるもの (説明変数 X)", 
            feature_candidates, 
            default=feature_candidates[:2] if len(feature_candidates)>=2 else feature_candidates
        )

    # --- 分析実行ボタン ---
    if st.button("🚀 分析を開始する", type="primary", use_container_width=True):
        if not feature_vars:
            st.warning("要因（説明変数）を少なくとも1つ選んでください。")
        else:
            with st.spinner("AIが統計モデルを計算中..."):
                res = run_regression_analysis(df_numeric, target_var, feature_vars)
                # ★修正点: 結果をsession_stateに保存する
                st.session_state['res'] = res

    # --- 結果の表示処理 (session_stateに結果があれば表示) ---
    if 'res' in st.session_state:
        res = st.session_state['res']

        if res["status"] == "error":
            st.error(f"エラーが発生しました: {res['message']}")
        else:
            # 変数選択が変わっていた場合の整合性チェック
            # (以前の結果と現在の変数が食い違っている場合、再実行を促すかエラー回避)
            if res['target'] != target_var or set(res['features']) != set(feature_vars):
                 st.warning("⚠️ 変数の選択が変更されました。「分析を開始する」ボタンをもう一度押して更新してください。")
            else:
                # =========================================
                # 結果表示パート
                # =========================================
                st.divider()
                st.header("📊 分析レポート")

                # --- 1. モデル精度 ---
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("決定係数 (R²)", f"{res['r2']:.3f}", help="1に近いほど予測精度が高い（0.5以上ならまあまあ）")
                with col_m2:
                    st.metric("自由度調整済み R²", f"{res['adj_r2']:.3f}", help="変数の数を考慮した精度。より厳密な指標。")
                with col_m3:
                    score = res['r2']
                    if score > 0.8: eval_text = "🌟 非常に高い精度です！"
                    elif score > 0.5: eval_text = "✅ 信頼できる精度です"
                    else: eval_text = "⚠️ 精度は低めです（他の要因が必要かも）"
                    st.info(f"**AI判定:**\n\n{eval_text}")

                # タブ切り替え
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🏆 要因の影響度ランキング", 
                    "📝 AI詳細解説", 
                    "🔮 未来シミュレーター", 
                    "📈 診断グラフ"
                ])

                # === Tab 1: 影響度ランキング ===
                with tab1:
                    st.subheader("結局、何が一番重要なのか？")
                    st.markdown("単位を無視して、**「純粋な影響力の強さ」**を比較したグラフです。")
                    
                    res_df = res["result_df"].copy()
                    res_df["abs_impact"] = res_df["標準化係数 (影響度)"].abs()
                    res_df = res_df.sort_values("abs_impact", ascending=True)

                    res_df["color"] = res_df["標準化係数 (影響度)"].apply(lambda x: "プラスの影響 (増える)" if x > 0 else "マイナスの影響 (減る)")

                    fig_bar = px.bar(
                        res_df, 
                        x="標準化係数 (影響度)", 
                        y="変数名", 
                        orientation='h',
                        color="color",
                        color_discrete_map={"プラスの影響 (増える)": "#3366CC", "マイナスの影響 (減る)": "#DC3912"},
                        text_auto=".2f",
                        title=f"「{target_var}」への影響度ランキング"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.caption("※ 棒が長いほど、結果に対する支配力が強い要因です。")

                # === Tab 2: AI詳細解説 ===
                with tab2:
                    st.subheader("🧐 各要因の詳細評価")
                    display_df = res["result_df"].drop(columns=["abs_impact", "color"], errors='ignore')
                    
                    for index, row in display_df.iterrows():
                        with st.expander(f"📌 **{row['変数名']}** の評価", expanded=True):
                            c1, c2, c3 = st.columns([1, 1, 2])
                            is_significant = row['P値 (信頼度)'] < 0.05
                            sig_icon = "✅" if is_significant else "❓"
                            sig_text = "統計的に信頼できます" if is_significant else "偶然の可能性があります"
                            
                            with c1:
                                st.metric("1増えるとどうなる？", f"{row['係数 (傾き)']:.2f}")
                            with c2:
                                st.metric("信頼性", sig_icon, help=f"P値: {row['P値 (信頼度)']:.4f}")
                                st.caption(sig_text)
                            with c3:
                                impact_dir = "増加" if row['係数 (傾き)'] > 0 else "減少"
                                st.markdown(f"""
                                **【AI解説】**
                                この変数が **1** 増えると、{target_var}は約 **{abs(row['係数 (傾き)']):.2f} {impact_dir}** すると予測されます。
                                """)

                # === Tab 3: 未来シミュレーター ===
                with tab3:
                    st.subheader("🎛️ もし条件を変えたらどうなる？")
                    st.markdown("スライダーを動かして、未来の結果を予測してみましょう。")
                    
                    user_inputs = {}
                    col_sim = st.columns(2)
                    
                    # スライダーの再描画によるリセットを防ぐため、session_stateはここで活きる
                    for i, feature in enumerate(feature_vars):
                        min_val = float(res['data'][feature].min())
                        max_val = float(res['data'][feature].max())
                        mean_val = float(res['data'][feature].mean())
                        
                        with col_sim[i % 2]:
                            user_inputs[feature] = st.slider(
                                f"🎚️ {feature}", 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=mean_val,
                                key=f"sim_slider_{feature}" # キーを一意にする
                            )

                    const = res['model'].params['const']
                    prediction = const
                    
                    for feature, value in user_inputs.items():
                        coef = res['result_df'][res['result_df']['変数名'] == feature]['係数 (傾き)'].values[0]
                        prediction += coef * value
                    
                    st.markdown("---")
                    st.markdown(f"### 🎯 予測される {target_var}")
                    st.markdown(f"# **{prediction:,.1f}**")

                # === Tab 4: 診断グラフ ===
                with tab4:
                    st.subheader("📈 予測精度と残差のチェック")
                    pred_y = res['model'].predict(sm.add_constant(res['data'][feature_vars]))
                    actual_y = res['data'][target_var]
                    
                    fig_sc = px.scatter(
                        x=actual_y, y=pred_y, 
                        labels={'x': '実際の結果', 'y': 'AIの予測値'},
                        title="予測の答え合わせ"
                    )
                    min_all = min(actual_y.min(), pred_y.min())
                    max_all = max(actual_y.max(), pred_y.max())
                    fig_sc.add_shape(type="line", x0=min_all, y0=min_all, x1=max_all, y1=max_all,
                                    line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_sc, use_container_width=True)

if __name__ == "__main__":
    main()