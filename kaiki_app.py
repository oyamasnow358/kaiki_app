import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import io

# --- 1. 初期設定 ---
st.set_page_config(
    page_title="要因分析・未来予測アプリ",
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
    """重回帰分析の実行ロジック"""
    try:
        # データの準備
        data = df[[target_col] + feature_cols].dropna()
        if len(data) < len(feature_cols) + 2:
            return {"status": "error", "message": "データ数が少なすぎます。"}

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

        return {
            "status": "success",
            "model": model,
            "result_df": result_df,
            "r2": model.rsquared,
            "adj_r2": model.rsquared_adj,
            "data": data,
            "target": target_col,
            "features": feature_cols
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 3. メインアプリ ---

def main():
    st.title("🚀 要因分析・未来予測アプリ")
    st.markdown("""
    「結果」を変えるための**「重要な要因」**を見つけ、条件を変えたときの**「未来」**をシミュレーションします。
    """)
    
    # --- サイドバー ---
    with st.sidebar:
        st.header("📂 データ設定")
        uploaded_file = st.file_uploader("CSVをアップロード", type=["csv"])
        
        st.markdown("---")
        st.markdown("##### 📌 テスト用データ")
        csv_text = create_csv_template()
        st.download_button("📥 サンプルCSV", csv_text.encode('utf-8-sig'), "sample_regression.csv", "text/csv")

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
        st.error("分析には数値の列が2つ以上必要です。")
        return

    # --- 変数選択 ---
    st.markdown("### 1. 何を分析しますか？")
    col_var1, col_var2 = st.columns(2)
    
    with col_var1:
        target_var = st.selectbox("🎯 予測・改善したい結果 (Y)", df_numeric.columns, index=0)
    
    with col_var2:
        cands = [c for c in df_numeric.columns if c != target_var]
        feature_vars = st.multiselect(
            "⚡ 要因と思われるもの (X)", 
            cands, 
            default=cands[:2] if len(cands)>=2 else cands
        )

    # --- 実行ボタン ---
    if st.button("🚀 分析を開始する", type="primary", use_container_width=True):
        if not feature_vars:
            st.warning("要因を少なくとも1つ選んでください。")
        else:
            with st.spinner("計算中..."):
                res = run_regression_analysis(df_numeric, target_var, feature_vars)
                st.session_state['reg_res'] = res

    # --- 結果表示 ---
    if 'reg_res' in st.session_state:
        res = st.session_state['reg_res']

        if res["status"] == "error":
            st.error(f"エラー: {res['message']}")
        else:
            if res['target'] != target_var or set(res['features']) != set(feature_vars):
                 st.warning("⚠️ 選択項目が変わりました。再度ボタンを押してください。")
            else:
                st.divider()

                # --- 4つのタブ構成（ご希望の形） ---
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🏆 影響度ランキング", 
                    "🧐 要因ごとの詳細評価", 
                    "🔮 未来シミュレーション", 
                    "📈 精度とデータの確認"
                ])

                # === Tab 1: 影響度ランキング ===
                with tab1:
                    st.subheader("結局、何が一番効くのか？")
                    st.markdown("単位（円や分など）を無視して、**「純粋な影響力の強さ」**だけを比較したグラフです。")
                    
                    res_df = res["result_df"].copy()
                    res_df["abs_impact"] = res_df["標準化係数 (影響度)"].abs()
                    res_df = res_df.sort_values("abs_impact", ascending=True)

                    res_df["color"] = res_df["標準化係数 (影響度)"].apply(
                        lambda x: "青: 増やすと結果が良くなる" if x > 0 else "赤: 増やすと結果が悪くなる"
                    )

                    fig_bar = px.bar(
                        res_df, 
                        x="標準化係数 (影響度)", y="変数名", 
                        orientation='h', color="color",
                        color_discrete_map={"青: 増やすと結果が良くなる": "#3366CC", "赤: 増やすと結果が悪くなる": "#DC3912"},
                        text_auto=".2f",
                        title=f"「{target_var}」への影響力ランキング"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.caption("棒が長いほど、結果を支配する力が強い「重要な要因」です。")

                # === Tab 2: 要因ごとの詳細評価（AI解説を撤廃） ===
                with tab2:
                    st.subheader("数値を詳しく見る")
                    st.markdown("各要因が**「具体的にどれくらい結果を変えるか」**と**「その数値は信頼できるか」**の判定です。")
                    
                    display_df = res["result_df"].sort_values("標準化係数 (影響度)", key=abs, ascending=False)
                    
                    for index, row in display_df.iterrows():
                        with st.expander(f"📌 **{row['変数名']}** の評価", expanded=True):
                            c1, c2, c3 = st.columns([1, 1, 2])
                            
                            is_reliable = row['P値 (信頼度)'] < 0.05
                            icon = "✅" if is_reliable else "❓"
                            reliability_text = "統計的に信頼できます" if is_reliable else "偶然の可能性があります"
                            
                            with c1:
                                st.metric("1増えると？", f"{row['係数 (傾き)']:.2f}", help="実際の単位での変化量")
                            with c2:
                                st.metric("信頼性判定", icon, help=f"P値: {row['P値 (信頼度)']:.4f}")
                                st.caption(reliability_text)
                            with c3:
                                action = "増やす" if row['係数 (傾き)'] > 0 else "減らす"
                                direction = "増え" if row['係数 (傾き)'] > 0 else "減り"
                                
                                # 言葉の修正：AI解説 → ポイント解説
                                st.markdown("**【ポイント解説】**")
                                if is_reliable:
                                    st.success(f"これを **1** {action}と、{target_var}は約 **{abs(row['係数 (傾き)']):.2f} {direction}ます**。\n確かな要因と言えます。")
                                else:
                                    st.warning(f"計算上は **{abs(row['係数 (傾き)']):.2f} {direction}** と出ましたが、\nデータのバラつきが大きく、**断定できません**。参考程度にしてください。")

                # === Tab 3: 未来シミュレーション ===
                with tab3:
                    st.subheader("🎛️ もし条件を変えたらどうなる？")
                    st.markdown("スライダーを動かすと、下の予測値がリアルタイムで変わります。")
                    
                    user_inputs = {}
                    col_sim = st.columns(2)
                    
                    # session_stateのおかげで、スライダーを動かしてもリセットされません
                    for i, feature in enumerate(feature_vars):
                        min_val = float(res['data'][feature].min())
                        max_val = float(res['data'][feature].max())
                        mean_val = float(res['data'][feature].mean())
                        
                        with col_sim[i % 2]:
                            user_inputs[feature] = st.slider(
                                f"🎚️ {feature}", 
                                min_value=min_val, max_value=max_val, value=mean_val,
                                key=f"sim_{feature}"
                            )

                    const = res['model'].params['const']
                    prediction = const
                    for feature, value in user_inputs.items():
                        coef = res['result_df'][res['result_df']['変数名'] == feature]['係数 (傾き)'].values[0]
                        prediction += coef * value
                    
                    st.markdown("---")
                    st.markdown(f"### 🎯 予測される {target_var}")
                    st.markdown(f"# **{prediction:,.1f}**")
                    st.info("※ Tab 2で「信頼性判定 ✅」が出ている項目を動かした時のみ、この予測は信用できます。")

                # === Tab 4: 精度とデータの確認 ===
                with tab4:
                    st.subheader("📈 予測の精度チェック")
                    
                    r2 = res['r2']
                    col_chk1, col_chk2 = st.columns(2)
                    with col_chk1:
                        st.metric("モデルの精度 (決定係数)", f"{r2*100:.1f}%")
                    with col_chk2:
                        if r2 > 0.8: st.success("非常に高い精度です。よく当てはまっています。")
                        elif r2 > 0.5: st.info("まあまあの精度です。傾向はつかめます。")
                        else: st.error("精度が低いです。他の重要な要因が抜けているかもしれません。")

                    st.markdown("#### 実測値 vs 予測値")
                    pred_y = res['model'].predict(sm.add_constant(res['data'][feature_vars]))
                    actual_y = res['data'][target_var]
                    
                    fig_sc = px.scatter(
                        x=actual_y, y=pred_y, 
                        labels={'x': '実際の結果', 'y': '計算上の予測値'},
                        title="答え合わせ (点線に近いほど正確)"
                    )
                    min_all = min(actual_y.min(), pred_y.min())
                    max_all = max(actual_y.max(), pred_y.max())
                    fig_sc.add_shape(type="line", x0=min_all, y0=min_all, x1=max_all, y1=max_all,
                                    line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_sc, use_container_width=True)
                    
                    st.markdown("#### 使用データ")
                    st.dataframe(res['data'])

if __name__ == "__main__":
    main()