import streamlit as st

# セッション状態の初期化
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_method" not in st.session_state:
    st.session_state.selected_method = None

# ページ遷移関数：セッション状態を更新後、即時再レンダリング
def go_to_page(page_name):
    st.session_state.page = page_name
    st.experimental_rerun()

# ホームページ
def home():
    st.title("特別支援教育ツール")
    st.write("このツールでは、子供の困り感に応じた支援方法を探すことができます。")

    # 困り感の選択
    st.subheader("どんな困り感がありますか？")
    difficulties = [
        "落ち着きがない", 
        "コミュニケーションが苦手", 
        "自己管理ができない", 
        "感情をコントロールできない", 
        "学習面で遅れがある"
    ]
    selected_difficulty = st.selectbox("困り感を選択してください:", difficulties)

    if selected_difficulty:
        st.write(f"選択した困り感: **{selected_difficulty}**")
        if st.button("適した方法を確認する"):
            go_to_page("analysis_methods")

    # 分析方法一覧へのリンク
    st.subheader("使える分析方法の一覧")
    if st.button("分析方法一覧を見る"):
        go_to_page("analysis_methods")

# 分析方法の一覧ページ
def analysis_methods():
    st.title("分析方法の一覧")
    st.write("以下の方法から選択して、詳細をご覧ください。")
    methods = ["応用行動分析 (ABA)", "認知行動療法 (CBT)", "感覚統合療法", "その他"]

    for method in methods:
        if st.button(f"➡ {method}", key=f"btn_{method}"):
            st.session_state.selected_method = method
            go_to_page("method_details")

    if st.button("⬅ ホームに戻る"):
        go_to_page("home")

# 各方法の詳細ページ
def method_details():
    method = st.session_state.get("selected_method", "方法が選択されていません")
    st.title(f"{method} の詳細情報")

    # どのような実態の子供に使えるか
    st.subheader("どのような実態の子供に使えるか")
    if method == "応用行動分析 (ABA)":
        st.info("ABAは、行動面の課題を持つ子供に特に有効です。反復行動や適応スキルの不足が見られる場合に適用されます。")
    elif method == "認知行動療法 (CBT)":
        st.info("CBTは、不安、抑うつ、衝動性の課題を持つ子供に効果的です。感情調整スキル向上を目指します。")
    elif method == "感覚統合療法":
        st.info("感覚過敏や鈍感など、感覚処理の課題を持つ子供に有効です。")
    else:
        st.info("情報がありません。")

    # 具体的な使い方
    st.subheader("具体的な使い方")
    if method == "応用行動分析 (ABA)":
        st.image("https://via.placeholder.com/600x300", caption="ABAの実施例（図はサンプル）")
        st.write("1. ターゲット行動を特定します。\n2. 行動を強化するための報酬を設定します。\n3. 反復的な練習を通じて行動を学びます。\n\nABAでは、行動を定量的に評価し、進捗を明確に把握します。")
    elif method == "認知行動療法 (CBT)":
        st.image("https://via.placeholder.com/600x300", caption="CBTでのセッション例（図はサンプル）")
        st.write("1. 子供の思考パターンを一緒に確認します。\n2. 否定的な思考をポジティブな思考に置き換える練習を行います。\n3. ワークシートを用いて繰り返し実践します。\n\nCBTは感情の認識や管理スキルの向上を目指します。")
    elif method == "感覚統合療法":
        st.image("https://via.placeholder.com/600x300", caption="感覚統合療法の実施例（図はサンプル）")
        st.write("1. 感覚刺激に応じた活動を設定します（例：バランスボール、触覚素材）。\n2. 子供が安心して取り組める環境を整えます。\n3. 繰り返しの活動を通じて感覚処理能力を高めます。")

    if st.button("⬅ 分析方法一覧に戻る"):
        go_to_page("analysis_methods")

# ページの分岐
if st.session_state.page == "home":
    home()
elif st.session_state.page == "analysis_methods":
    analysis_methods()
elif st.session_state.page == "method_details":
    method_details()
