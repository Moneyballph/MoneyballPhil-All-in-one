import streamlit as st
import base64
from pathlib import Path

def set_background(img_rel_path: str = "assets/mbp_bg.png"):
    """
    Looks for the background image in multiple sensible places:
    - as given (img_rel_path)
    - relative to this file's folder
    - '<file_dir>/assets/mbp_bg.png'
    - '<file_dir>/mbp_bg.png'
    Falls back to black if not found.
    """
    file_dir = Path(__file__).parent.resolve()
    candidates = [
        Path(img_rel_path),                        # as provided
        file_dir / img_rel_path,                   # relative to this script
        file_dir / "assets" / "mbp_bg.png",
        file_dir / "mbp_bg.png",
    ]

    img_path = next((p for p in candidates if p.exists()), None)

    if not img_path:
        st.warning("⚠️ Background image not found. Checked: " +
                   ", ".join(str(p) for p in candidates) +
                   ". Using plain black.")
        st.markdown("<style>.stApp{background:#000}</style>", unsafe_allow_html=True)
        return

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{b64}") center/cover no-repeat fixed;
        }}
        .stApp::before {{
            content:""; position:fixed; inset:0;
            background: rgba(0,0,0,0.18); z-index:-1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# call once, near the top of your file
set_background("assets/mbp_bg.png")


# =========================
# Hitter Hit Probability Sim
# =========================
def run_hitter_simulator():
    st.title("💰 Moneyball Phil: Hit Probability Simulator")

    st.header("📥 Player Stat Entry")
    with st.form("player_input"):
        name = st.text_input("Player Name")
        season_avg = st.number_input("Season AVG", 0.0, 1.0, step=0.0001, format="%.4f")
        last7_avg = st.number_input("Last 7 Days AVG", 0.0, 1.0, step=0.0001, format="%.4f")
        split_avg = st.number_input("Split AVG (Home/Away)", 0.0, 1.0, step=0.0001, format="%.4f")
        hand_avg = st.number_input("AVG vs Handedness", 0.0, 1.0, step=0.0001, format="%.4f")
        pitcher_avg = st.number_input("AVG vs Pitcher", 0.0, 1.0, step=0.0001, format="%.4f")
        ab_vs_pitcher = st.number_input("At-Bats vs Pitcher", min_value=0, step=1)
        pitcher_hand = st.selectbox("Pitcher Handedness", ["Right", "Left"])
        batting_order = st.selectbox("Batting Order Position", list(range(1, 10)))
        odds = st.number_input("Sportsbook Odds (American)", step=1)
        pitcher_era = st.number_input("Pitcher ERA", 0.0, 15.0, step=0.01)
        pitcher_whip = st.number_input("Pitcher WHIP", 0.0, 3.0, step=0.01)
        pitcher_k9 = st.number_input("Pitcher K/9", min_value=0.0, step=0.1)
        submit = st.form_submit_button("Simulate Player")

    if "players" not in st.session_state:
        st.session_state.players = []

    def calculate_weighted_avg(season, last7, split, hand, pitcher):
        return round(0.2 * season + 0.3 * last7 + 0.2 * split + 0.2 * hand + 0.1 * pitcher, 4)

    def binomial_hit_probability(avg, ab=4):
        return round(1 - (1 - avg)**ab, 4)

    def american_to_implied(odds):
        return round(abs(odds) / (abs(odds) + 100), 4) if odds < 0 else round(100 / (odds + 100), 4)

    def classify_zone(prob):
        if prob >= 0.8:
            return "🟩 Elite"
        elif prob >= 0.7:
            return "🟨 Strong"
        elif prob >= 0.6:
            return "🟧 Moderate"
        else:
            return "🟥 Risky"

    if submit:
        weighted_avg = calculate_weighted_avg(season_avg, last7_avg, split_avg, hand_avg, pitcher_avg)

        # Pitcher Difficulty Adjustment
        if pitcher_whip >= 1.40 or pitcher_era >= 5.00:
            adjustment = 0.020
            tier = "🟢 Easy Pitcher"
        elif pitcher_whip < 1.10 or pitcher_era < 3.50:
            adjustment = -0.020
            tier = "🔴 Tough Pitcher"
        else:
            adjustment = 0.000
            tier = "🟨 Average Pitcher"

        adj_weighted_avg = round(weighted_avg + adjustment, 4)
        st.markdown(f"**Weighted AVG (Before Adjustment):** `{weighted_avg}`")
        st.markdown(f"**Pitcher Difficulty Tier:** {tier} (`{adjustment:+.3f}`)")
        st.markdown(f"**Adjusted Weighted AVG:** `{adj_weighted_avg}`")

        ab_lookup = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.2, 6: 4.0, 7: 3.8, 8: 3.6, 9: 3.4}
        est_ab = ab_lookup.get(batting_order, 4.0)
        st.markdown(f"**Estimated At-Bats:** {est_ab} (based on batting {batting_order}th)")

        true_prob = binomial_hit_probability(adj_weighted_avg, ab=round(est_ab))
        implied_prob = american_to_implied(odds)
        ev = round((true_prob - implied_prob) * 100, 1)
        zone = classify_zone(true_prob)

        st.session_state.players.append({
            "name": name,
            "true_prob": true_prob,
            "implied_prob": implied_prob,
            "ev": ev,
            "zone": zone
        })

    st.header("🔥 Top Hit Board")
    if st.session_state.players:
        df = pd.DataFrame(st.session_state.players)
        df = df.sort_values(by=["true_prob", "ev"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)
        df["True Prob %"] = (df["true_prob"] * 100).round(1)
        df["Implied %"] = (df["implied_prob"] * 100).round(1)
        df["EV %"] = df["ev"]
        st.dataframe(df[["Rank", "name", "True Prob %", "Implied %", "EV %", "zone"]], use_container_width=True)
    else:
        st.info("Add players above to populate the Top Hit Board.")

    st.header("🧮 Parlay Builder")
    if len(st.session_state.players) >= 2:
        names = [p["name"] for p in st.session_state.players]
        selected = st.multiselect("Select 2 or 3 Players", names)
        if len(selected) in [2, 3]:
            probs = [p["true_prob"] for p in st.session_state.players if p["name"] in selected]
            parlay_true = round(pd.Series(probs).prod(), 4)

            st.markdown("### 📉 Sportsbook Parlay Odds Input")
            parlay_odds = st.number_input("Enter Combined Parlay Odds (American)", step=1, key="parlay_odds")
            implied_parlay_prob = (abs(parlay_odds) / (abs(parlay_odds) + 100)) if parlay_odds < 0 else (100 / (parlay_odds + 100))
            parlay_ev = round((parlay_true - implied_parlay_prob) * 100, 1)

            if parlay_true >= 0.75:
                pzone = "🟩 Elite Parlay"
            elif parlay_true >= 0.60:
                pzone = "🟨 Strong Parlay"
            elif parlay_true >= 0.45:
                pzone = "🟧 Moderate Parlay"
            else:
                pzone = "🟥 Risky Parlay"

            st.markdown(f"**True Parlay Probability:** {parlay_true:.2%}")
            st.markdown(f"**Implied Parlay Probability:** {implied_parlay_prob:.2%}")
            st.markdown(f"**Parlay EV %:** {parlay_ev:+.1f}%")
            st.markdown(f"**Parlay Zone:** {pzone}")
        elif len(selected) > 3:
            st.warning("Please select only 2 or 3 players.")
    else:
        st.info("Add at least 2 players to use the Parlay Builder.")

# =========================
# Pitcher Earned Runs Sim
# =========================
def run_pitcher_er_simulator():
    st.title("🎯 Pitcher Earned Runs Simulator")

    col1, col2 = st.columns(2)

    with col1:
        pitcher_name = st.text_input("Pitcher Name")
        era = st.number_input("ERA", value=3.50, step=0.01)
        total_ip = st.number_input("Total Innings Pitched", value=90.0, step=0.1)
        games_started = st.number_input("Games Started", value=15, step=1)
        last_3_ip = st.text_input("Last 3 Game IP (comma-separated, e.g. 5.2,6.1,5.0)")
        xera = st.number_input("xERA (optional, overrides ERA)", value=0.0, step=0.01)
        whip = st.number_input("WHIP (optional)", value=0.0, step=0.01)

    with col2:
        opponent_ops = st.number_input("Opponent OPS", value=0.670, step=0.001)
        league_avg_ops = st.number_input("League Average OPS", value=0.715, step=0.001)
        ballpark = st.selectbox("Ballpark Factor", ["Neutral", "Pitcher-Friendly", "Hitter-Friendly"])
        under_odds = st.number_input("Sportsbook Odds (U2.5 ER)", value=-115)
        simulate_button = st.button("▶ Simulate Player")

    if simulate_button:
        try:
            ip_values = [float(i.strip()) for i in last_3_ip.split(",") if i.strip() != ""]
            trend_ip = sum(ip_values) / len(ip_values)
        except:
            st.error("⚠️ Please enter 3 valid IP values separated by commas (e.g. 5.2,6.1,5.0)")
            st.stop()

        base_ip = total_ip / games_started

        if ballpark == "Pitcher-Friendly":
            park_adj = 0.2
        elif ballpark == "Hitter-Friendly":
            park_adj = -0.2
        else:
            park_adj = 0.0

        expected_ip = round(((base_ip + trend_ip) / 2) + park_adj, 2)

        used_era = xera if xera > 0 else era
        adjusted_era = round(used_era * (opponent_ops / league_avg_ops), 3)
        lambda_er = round(adjusted_era * (expected_ip / 9), 3)

        p0 = poisson.pmf(0, lambda_er)
        p1 = poisson.pmf(1, lambda_er)
        p2 = poisson.pmf(2, lambda_er)
        true_prob = round(p0 + p1 + p2, 4)

        if under_odds < 0:
            implied_prob = round(abs(under_odds) / (abs(under_odds) + 100), 4)
        else:
            implied_prob = round(100 / (under_odds + 100), 4)

        ev = round((true_prob - implied_prob) / implied_prob * 100, 2)

        if under_odds < 0:
            payout = 100 / abs(under_odds)
        else:
            payout = under_odds / 100
        true_ev = round((true_prob * payout) - ((1 - true_prob) * 1), 4)
        true_ev_percent = round(true_ev * 100, 2)

        if true_prob >= 0.80:
            tier = "🟢 Elite"
        elif true_prob >= 0.70:
            tier = "🟡 Strong"
        elif true_prob >= 0.60:
            tier = "🟠 Moderate"
        else:
            tier = "🔴 Risky"

        warning_msg = ""
        if whip > 1.45 and era < 3.20 and xera == 0:
            warning_msg = "⚠️ ERA may be misleading due to high WHIP. Consider using xERA or reducing confidence."

        st.subheader("📊 Simulation Results")
        st.markdown(f"**Expected IP:** {expected_ip} innings")
        st.markdown(f"**Adjusted ERA vs Opponent:** {adjusted_era}")
        st.markdown(f"**Expected ER (λ):** {lambda_er}")
        st.markdown(f"**True Probability of Under 2.5 ER:** {true_prob*100}%")
        st.markdown(f"**Implied Probability (from Odds):** {implied_prob*100}%")
        st.markdown(f"**Expected Value (EV%):** {ev}%")
        st.markdown(f"**True Expected Value (ROI per $1):** {true_ev_percent}%")
        st.markdown(f"**Difficulty Tier:** {tier}")
        if warning_msg:
            st.warning(warning_msg)

        df = pd.DataFrame({
            "Pitcher": [pitcher_name],
            "True Probability": [f"{true_prob*100:.1f}%"],
            "Implied Probability": [f"{implied_prob*100:.1f}%"],
            "EV %": [f"{ev:.1f}%"],
            "True EV %": [f"{true_ev_percent:.1f}%"],
            "Tier": [tier.replace("🟢 ", "").replace("🟡 ", "").replace("🟠 ", "").replace("🔴 ", "")]
        })
        st.dataframe(df, use_container_width=True)

# =========================
# NFL Prop Simulator (QB/WR/RB)
# =========================
def run_nfl_simulator():
    st.title("🏈 Moneyball Phil: NFL Prop Simulator (v2.0)")
    position = st.selectbox("Select Position", ["Quarterback", "Wide Receiver", "Running Back"])

    if "all_props" not in st.session_state:
        st.session_state.all_props = []

    # Helper functions (as provided)
    def implied_prob(odds):
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    def ev_calc(true_prob, odds):
        imp_prob = implied_prob(odds)
        return round((true_prob - imp_prob) * 100, 2)

    def get_tier(prob):
        if prob >= 80:
            return "🟢 Elite"
        elif prob >= 65:
            return "🟡 Strong"
        elif prob >= 50:
            return "🟠 Moderate"
        else:
            return "🔴 Risky"

    def logistic_prob(x, line, scale=15):
        return round((1 / (1 + math.exp(-(x - line) / scale))) * 100, 2)

    def classify_def_tier(val):
        if isinstance(val, int) or isinstance(val, float):
            if val < 205:
                return "🔴 Tough"
            elif val <= 240:
                return "🟡 Average"
            else:
                return "🟢 Easy"
        return "Unknown"

    # Quarterback Module
    if position == "Quarterback":
        st.header("🎯 Quarterback Inputs")
        name = st.text_input("Quarterback Name", value="")
        opp = st.text_input("Opponent Team", value="")

        st.subheader("📊 Passing Yards Props")
        std_line = st.number_input("Standard Passing Yards Line", value=0.0)
        over_std = st.number_input("Odds Over (Standard)", value=0.0)
        under_std = st.number_input("Odds Under (Standard)", value=0.0)
        alt_line = st.number_input("Alt Over Line", value=0.0)
        alt_odds = st.number_input("Odds for Alt Over", value=0.0)

        st.subheader("🎯 Touchdown Props")
        td_line = st.number_input("Passing TD Line", value=1.5)
        td_under_odds = st.number_input("Odds for Under TDs", value=0.0)

        st.subheader("📈 QB & Defense Stats")
        ypg = st.number_input("QB Yards/Game", value=0.0)
        tds = st.number_input("QB TD/Game", value=0.0)
        attempts = st.number_input("Pass Attempts/Game", value=0.0)
        def_yds = st.number_input("Defense Yards Allowed/Game", value=0.0)
        def_tds = st.number_input("Defense Pass TDs/Game", value=0.0)

        if st.button("Simulate QB Props"):
            std_prob = logistic_prob(ypg, std_line)
            alt_prob = logistic_prob(ypg, alt_line)
            td_prob = logistic_prob(tds, td_line, scale=0.5)
            tier = classify_def_tier(def_yds)

            st.success(f"📈 Standard Yards Hit %: {std_prob}%  | Alt Line %: {alt_prob}%")
            st.success(f"📉 Under {td_line} TDs Hit %: {100 - td_prob}%")
            st.info(f"Opponent Defense Tier: {tier}")

            st.session_state.all_props.extend([
                {"Player": name, "Prop": f"Over {std_line} Pass Yds", "True Prob": std_prob, "Odds": over_std},
                {"Player": name, "Prop": f"Over {alt_line} Alt Pass Yds", "True Prob": alt_prob, "Odds": alt_odds},
                {"Player": name, "Prop": f"Under {td_line} Pass TDs", "True Prob": 100 - td_prob, "Odds": td_under_odds},
            ])

    # Wide Receiver Module
    if position == "Wide Receiver":
        st.header("🎯 Wide Receiver Inputs")
        name = st.text_input("Wide Receiver Name", value="")
        opp = st.text_input("Opponent Team", value="")

        st.subheader("📊 Receiving Yards Props")
        std_line = st.number_input("Standard Receiving Yards Line", value=0.0)
        over_std = st.number_input("Odds Over (Standard)", value=0.0)
        under_std = st.number_input("Odds Under (Standard)", value=0.0)
        alt_line = st.number_input("Alt Over Line", value=0.0)
        alt_odds = st.number_input("Odds for Alt Over", value=0.0)

        st.subheader("🎯 Receptions Prop")
        rec_line = st.number_input("Receptions Line", value=0.0)
        rec_over_odds = st.number_input("Odds for Over Receptions", value=0.0)
        rec_under_odds = st.number_input("Odds for Under Receptions", value=0.0)

        st.subheader("📈 WR & Defense Stats")
        ypg = st.number_input("WR Yards/Game", value=0.0)
        rpg = st.number_input("WR Receptions/Game", value=0.0)
        def_yds = st.number_input("Defense WR Yards Allowed/Game", value=0.0)
        def_rec = st.number_input("Defense WR Receptions Allowed/Game", value=0.0)

        if st.button("Simulate WR Props"):
            std_prob = logistic_prob(ypg, std_line)
            alt_prob = logistic_prob(ypg, alt_line)
            rec_prob = logistic_prob(rpg, rec_line, scale=1.5)
            tier = classify_def_tier(def_yds)

            st.success(f"📈 Standard Yards Hit %: {std_prob}%  | Alt Line %: {alt_prob}%")
            st.success(f"🎯 Receptions Over {rec_line} Hit %: {rec_prob}%")
            st.success(f"📉 Receptions Under {rec_line} Hit %: {100 - rec_prob}%")
            st.info(f"Opponent Defense Tier: {tier} | Avg Receptions Allowed: {def_rec}")

            st.session_state.all_props.extend([
                {"Player": name, "Prop": f"Over {std_line} Rec Yds", "True Prob": std_prob, "Odds": over_std},
                {"Player": name, "Prop": f"Under {std_line} Rec Yds", "True Prob": 100 - std_prob, "Odds": under_std},
                {"Player": name, "Prop": f"Over {alt_line} Alt Rec Yds", "True Prob": alt_prob, "Odds": alt_odds},
                {"Player": name, "Prop": f"Over {rec_line} Receptions", "True Prob": rec_prob, "Odds": rec_over_odds},
                {"Player": name, "Prop": f"Under {rec_line} Receptions", "True Prob": 100 - rec_prob, "Odds": rec_under_odds},
            ])

    # Running Back Module
    if position == "Running Back":
        st.header("🎯 Running Back Inputs")
        name = st.text_input("Running Back Name", value="")
        opp = st.text_input("Opponent Team", value="")

        st.subheader("📊 Rushing Yards Props")
        std_line = st.number_input("Standard Rushing Yards Line", value=0.0)
        over_std = st.number_input("Odds Over (Standard)", value=0.0)
        under_std = st.number_input("Odds Under (Standard)", value=0.0)
        alt_line = st.number_input("Alt Over Line", value=0.0)
        alt_odds = st.number_input("Odds for Alt Over", value=0.0)

        st.subheader("🎯 Receptions Prop")
        rec_line = st.number_input("Receptions Line", value=0.0)
        rec_over_odds = st.number_input("Odds for Over Receptions", value=0.0)
        rec_under_odds = st.number_input("Odds for Under Receptions", value=0.0)

        st.subheader("📈 RB & Defense Stats")
        ypg = st.number_input("RB Yards/Game", value=0.0)
        rpg = st.number_input("RB Receptions/Game", value=0.0)
        def_yds = st.number_input("Defense Rush Yards Allowed/Game", value=0.0)
        def_rec = st.number_input("Defense RB Receptions Allowed/Game", value=0.0)

        if st.button("Simulate RB Props"):
            std_prob = logistic_prob(ypg, std_line)
            alt_prob = logistic_prob(ypg, alt_line)
            rec_prob = logistic_prob(rpg, rec_line, scale=1.5)
            tier = classify_def_tier(def_yds)

            st.success(f"📈 Standard Rush Yards Hit %: {std_prob}%  | Alt Line %: {alt_prob}%")
            st.success(f"🎯 Receptions Over {rec_line} Hit %: {rec_prob}%")
            st.success(f"📉 Receptions Under {rec_line} Hit %: {100 - rec_prob}%")
            st.info(f"Opponent Defense Tier: {tier} | Avg Receptions Allowed: {def_rec}")

            st.session_state.all_props.extend([
                {"Player": name, "Prop": f"Over {std_line} Rush Yds", "True Prob": std_prob, "Odds": over_std},
                {"Player": name, "Prop": f"Under {std_line} Rush Yds", "True Prob": 100 - std_prob, "Odds": under_std},
                {"Player": name, "Prop": f"Over {alt_line} Alt Rush Yds", "True Prob": alt_prob, "Odds": alt_odds},
                {"Player": name, "Prop": f"Over {rec_line} Receptions", "True Prob": rec_prob, "Odds": rec_over_odds},
                {"Player": name, "Prop": f"Under {rec_line} Receptions", "True Prob": 100 - rec_prob, "Odds": rec_under_odds},
            ])

    # Top Player Board
    st.markdown("---")
    st.subheader("📊 Top Player Board")
    if st.session_state.all_props:
        sorted_props = sorted(st.session_state.all_props, key=lambda x: x["True Prob"], reverse=True)
        for prop in sorted_props:
            ev = ev_calc(prop["True Prob"] / 100, prop["Odds"])
            tier = get_tier(prop["True Prob"])
            st.markdown(f"**{prop['Player']} – {prop['Prop']}**  ")
            st.markdown(f"True Prob: `{prop['True Prob']}%` | Odds: `{prop['Odds']}` | EV: `{ev}%` | Tier: {tier}", unsafe_allow_html=True)
    else:
        st.info("No props simulated yet. Run a player simulation to see results here.")

    # Parlay Builder
    st.markdown("---")
    st.subheader("💡 Parlay Builder")
    if len(st.session_state.all_props) >= 2:
        parlay_choices = [f"{p['Player']} – {p['Prop']}" for p in st.session_state.all_props]
        selected = st.multiselect("Select Props for Parlay (2+)", parlay_choices)

        if selected and len(selected) >= 2:
            selected_props = [p for p in st.session_state.all_props if f"{p['Player']} – {p['Prop']}" in selected]
            combined_prob = 1
            combined_ev = 0
            for p in selected_props:
                combined_prob *= p["True Prob"] / 100
                combined_ev += ev_calc(p["True Prob"] / 100, p["Odds"])
            combined_prob = round(combined_prob * 100, 2)
            avg_ev = round(combined_ev / len(selected_props), 2)
            st.success(f"Parlay Hit Probability: `{combined_prob}%` | Avg EV: `{avg_ev}%`")
    else:
        st.info("Add at least 2 simulated props to enable the parlay builder.")

# =========================
# Tabs Layout
# =========================
tabs = st.tabs([
    "⚾ Hitter Hit Probability",
    "⚾ Pitcher Earned Runs",
    "🏈 NFL Prop Simulator"
])

with tabs[0]:
    run_hitter_simulator()

with tabs[1]:
    run_pitcher_er_simulator()

with tabs[2]:
    run_nfl_simulator()



