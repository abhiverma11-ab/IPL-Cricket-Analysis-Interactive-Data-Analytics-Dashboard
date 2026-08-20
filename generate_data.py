"""
Generates a realistic synthetic IPL dataset for analysis purposes.

NOTE: This creates SAMPLE data modeled on real IPL structure (teams, seasons,
venues, toss dynamics, batting/bowling patterns) since this environment has no
internet access to pull an actual dataset (e.g. Kaggle's IPL ball-by-ball data).

If you have a real IPL dataset (e.g. matches.csv / deliveries.csv from Kaggle),
just drop it into the data/ folder with the same column names used here and
analysis.py will work on it unchanged.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru",
    "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad", "Gujarat Titans",
    "Lucknow Super Giants"
]

VENUES = [
    "Wankhede Stadium, Mumbai", "M. A. Chidambaram Stadium, Chennai",
    "M. Chinnaswamy Stadium, Bengaluru", "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi", "Narendra Modi Stadium, Ahmedabad",
    "Rajiv Gandhi International Stadium, Hyderabad", "Sawai Mansingh Stadium, Jaipur"
]

SEASONS = list(range(2018, 2025))

# --- TEAM STRENGTH -----------------------------------------------------
# Controls how likely each team is to win, independent of the toss.
# 1.0 = average team. Higher = stronger (wins more), lower = weaker.
# Change these numbers to make any team dominate or struggle more.
TEAM_STRENGTH = {
    "Mumbai Indians": 1.10,
    "Chennai Super Kings": 1.15,
    "Royal Challengers Bengaluru": 1.35,
    "Kolkata Knight Riders": 1.05,
    "Delhi Capitals": 0.85,
    "Punjab Kings": 0.80,
    "Rajasthan Royals": 1.00,
    "Sunrisers Hyderabad": 0.95,
    "Gujarat Titans": 1.15,
    "Lucknow Super Giants": 1.00,
}

# Assign a player pool per team: batsmen-leaning and bowler-leaning, some all-rounders
PLAYER_POOL = {}

# Generic, clearly-fictional Indian-style first/last names — deliberately avoids
# combining real cricketers' actual first + last names (the earlier version
# accidentally mashed up real players, e.g. "MS" + "Malik" or "Jasprit" + "Yadav",
# which looked like real player names with fake stats attached).
first_names = [
    "Arjun", "Vikram", "Rahul", "Aditya", "Karan", "Rohan", "Aryan", "Dev",
    "Kabir", "Rajat", "Sanjay", "Manish", "Nikhil", "Varun", "Aman", "Pranav",
    "Siddharth", "Harsh", "Yash", "Abhishek", "Gaurav", "Vivek", "Anand", "Sameer",
    "Naveen", "Karthik", "Ajay", "Ankit", "Ravi", "Deepak"
]
last_names = [
    "Verma", "Mehta", "Rathore", "Nair", "Bhatt", "Chauhan", "Desai", "Kapoor",
    "Bose", "Menon", "Trivedi", "Reddy", "Naidu", "Iyer", "Kulkarni", "Joshi",
    "Saxena", "Thakur", "Dubey", "Pillai", "Ghosh", "Rao", "Sinha", "Mishra",
    "Bhatia", "Suresh", "Prasad", "Chatterjee", "Krishnan", "Malhotra"
]

# Build a pool of unique full names (first x last combos), then hand them out
# to teams so no two players in the whole dataset share a name.
all_combos = [(f, l) for f in first_names for l in last_names]
rng.shuffle(all_combos)
NUM_PLAYERS_NEEDED = len(TEAMS) * 16
unique_names = [f"{f} {l}" for f, l in all_combos[:NUM_PLAYERS_NEEDED]]

# --- NAME OVERRIDES -----------------------------------------------------
# Assign specific (real) player names + role to a specific team's squad.
# Real, well-known IPL players are listed here (roster changes year to
# year — treat this as illustrative, not an official current squad list).
# All stats generated for them are still fully SYNTHETIC/simulated.
# Format: (name, role) — role is "batsman", "bowler", or "allrounder".
TEAM_NAME_OVERRIDES = {
    "Royal Challengers Bengaluru": [
        ("Virat Kohli", "batsman"), ("Faf du Plessis", "batsman"),
        ("Glenn Maxwell", "allrounder"), ("Mohammed Siraj", "bowler"),
        ("Rajat Patidar", "batsman"), ("Cameron Green", "allrounder"),
        ("Yash Dayal", "bowler"),
    ],
    "Mumbai Indians": [
        ("Rohit Sharma", "batsman"), ("Jasprit Bumrah", "bowler"),
        ("Suryakumar Yadav", "batsman"), ("Hardik Pandya", "allrounder"),
        ("Ishan Kishan", "batsman"), ("Tilak Varma", "batsman"),
        ("Piyush Chawla", "bowler"),
    ],
    "Chennai Super Kings": [
        ("MS Dhoni", "batsman"), ("Ravindra Jadeja", "allrounder"),
        ("Ruturaj Gaikwad", "batsman"), ("Deepak Chahar", "bowler"),
        ("Shivam Dube", "allrounder"), ("Matheesha Pathirana", "bowler"),
        ("Moeen Ali", "allrounder"),
    ],
    "Kolkata Knight Riders": [
        ("Shreyas Iyer", "batsman"), ("Andre Russell", "allrounder"),
        ("Sunil Narine", "allrounder"), ("Varun Chakravarthy", "bowler"),
        ("Rinku Singh", "batsman"), ("Venkatesh Iyer", "allrounder"),
        ("Mitchell Starc", "bowler"),
    ],
    "Delhi Capitals": [
        ("Rishabh Pant", "batsman"), ("Axar Patel", "allrounder"),
        ("Kuldeep Yadav", "bowler"), ("Mitchell Marsh", "allrounder"),
        ("Tristan Stubbs", "batsman"), ("Abishek Porel", "batsman"),
        ("Ishant Sharma", "bowler"),
    ],
    "Punjab Kings": [
        ("Shikhar Dhawan", "batsman"), ("Arshdeep Singh", "bowler"),
        ("Liam Livingstone", "allrounder"), ("Sam Curran", "allrounder"),
        ("Jitesh Sharma", "batsman"), ("Kagiso Rabada", "bowler"),
        ("Prabhsimran Singh", "batsman"),
    ],
    "Rajasthan Royals": [
        ("Sanju Samson", "batsman"), ("Yashasvi Jaiswal", "batsman"),
        ("Jos Buttler", "batsman"), ("Riyan Parag", "allrounder"),
        ("Yuzvendra Chahal", "bowler"), ("Trent Boult", "bowler"),
        ("Dhruv Jurel", "batsman"),
    ],
    "Sunrisers Hyderabad": [
        ("Pat Cummins", "bowler"), ("Heinrich Klaasen", "batsman"),
        ("Abhishek Sharma", "batsman"), ("Travis Head", "batsman"),
        ("Bhuvneshwar Kumar", "bowler"), ("Nitish Kumar Reddy", "allrounder"),
        ("T Natarajan", "bowler"),
    ],
    "Gujarat Titans": [
        ("Shubman Gill", "batsman"), ("Rashid Khan", "bowler"),
        ("Sai Sudharsan", "batsman"), ("Mohammed Shami", "bowler"),
        ("David Miller", "batsman"), ("Rahul Tewatia", "allrounder"),
        ("Gerald Coetzee", "bowler"),
    ],
    "Lucknow Super Giants": [
        ("Nicholas Pooran", "batsman"), ("Mayank Yadav", "bowler"),
        ("Ravi Bishnoi", "bowler"), ("Ayush Badoni", "batsman"),
        ("Marcus Stoinis", "allrounder"), ("Mohsin Khan", "bowler"),
        ("Quinton de Kock", "batsman"),
    ],
}

# --- PLAYER SKILL BOOST -----------------------------------------------
# Multiplies a named player's batting output (runs) relative to a normal
# player of the same role. 1.0 = no boost. Use this to make a specific
# player top the leaderboard, e.g. {"Virat Kohli": 1.9}.
PLAYER_SKILL_BOOST = {
    "Virat Kohli": 1.9,
}

reserved_names = set(name for players in TEAM_NAME_OVERRIDES.values() for name, _ in players)
unique_names = [n for n in unique_names if n not in reserved_names]
unique_names = unique_names[:NUM_PLAYERS_NEEDED]

name_pool_iter = iter(unique_names)
for team in TEAMS:
    squad = []
    overrides = TEAM_NAME_OVERRIDES.get(team, [])
    for i in range(16):
        if i < len(overrides):
            name, role = overrides[i]
        else:
            name = next(name_pool_iter)
            role = rng.choice(["batsman","bowler","allrounder","batsman","bowler"], p=[0.35,0.35,0.15,0.10,0.05])
        squad.append({"name": name, "role": role})
    PLAYER_POOL[team] = squad

matches = []
batting_rows = []
bowling_rows = []

match_id = 1
for season in SEASONS:
    season_teams = TEAMS  # all teams play each season here
    fixtures = []
    for i, t1 in enumerate(season_teams):
        for t2 in season_teams[i+1:]:
            fixtures.append((t1, t2))
    rng.shuffle(fixtures)
    fixtures = fixtures[:60]  # ~60 matches per season

    for (team1, team2) in fixtures:
        venue = rng.choice(VENUES)
        toss_winner = rng.choice([team1, team2])
        toss_decision = rng.choice(["bat", "field"], p=[0.35, 0.65])  # modern IPL bias to field first

        # Winner depends mainly on relative team strength, with the toss
        # giving a smaller secondary nudge (as it does in real cricket).
        # Strength is cubed before comparing so that realistic-looking
        # differences (e.g. 1.25 vs 0.80) translate into a clearly visible
        # gap in the final standings rather than getting lost in randomness.
        s1, s2 = TEAM_STRENGTH.get(team1, 1.0) ** 3, TEAM_STRENGTH.get(team2, 1.0) ** 3
        team1_strength_prob = s1 / (s1 + s2)

        toss_bonus = 0.06 if toss_decision == "field" else 0.03
        if toss_winner == team1:
            team1_win_prob = min(0.95, team1_strength_prob + toss_bonus)
        else:
            team1_win_prob = max(0.05, team1_strength_prob - toss_bonus)

        winner = team1 if rng.random() < team1_win_prob else team2

        player_of_match_team = winner
        matches.append({
            "match_id": match_id,
            "season": season,
            "venue": venue,
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "winner": winner
        })

        # Generate batting performances for both teams (top 8 batters who bat)
        for team in [team1, team2]:
            squad = PLAYER_POOL[team]
            batters = [p for p in squad if p["role"] in ("batsman", "allrounder")][:8]
            if len(batters) < 6:
                batters = squad[:8]
            for p in batters:
                boost = PLAYER_SKILL_BOOST.get(p["name"], 1.0)
                skill = (1.3 if p["role"] == "batsman" else 1.0) * boost
                balls = max(0, int(rng.normal(18 * skill, 10)))
                if balls <= 0:
                    continue
                sr_base = rng.normal(135 * boost, 20)
                runs = max(0, int(balls * (sr_base / 100) + rng.normal(0, 8)))
                fours = max(0, int(runs * rng.uniform(0.25, 0.45) / 4))
                sixes = max(0, int(runs * rng.uniform(0.15, 0.3) / 6))
                out = rng.random() < (0.72 / boost)
                batting_rows.append({
                    "match_id": match_id, "season": season, "team": team,
                    "player": p["name"], "runs": runs, "balls": balls,
                    "fours": fours, "sixes": sixes, "out": out
                })

        # Generate bowling performances for both teams (top 6 bowlers)
        for team in [team1, team2]:
            squad = PLAYER_POOL[team]
            bowlers = [p for p in squad if p["role"] in ("bowler", "allrounder")][:6]
            if len(bowlers) < 4:
                bowlers = squad[:6]
            for p in bowlers:
                skill = 1.2 if p["role"] == "bowler" else 1.0
                overs = round(rng.choice([2, 3, 4]) if rng.random() < 0.85 else rng.uniform(1, 4), 1)
                econ = max(4.5, rng.normal(8.3 / skill, 1.8))
                runs_conceded = max(0, int(overs * econ))
                wkt_prob = rng.random()
                wickets = 0
                if wkt_prob > 0.55:
                    wickets = 1
                if wkt_prob > 0.80:
                    wickets = 2
                if wkt_prob > 0.94:
                    wickets = 3
                if wkt_prob > 0.985:
                    wickets = 4
                bowling_rows.append({
                    "match_id": match_id, "season": season, "team": team,
                    "player": p["name"], "overs": overs, "runs_conceded": runs_conceded,
                    "wickets": wickets
                })

        match_id += 1

matches_df = pd.DataFrame(matches)
batting_df = pd.DataFrame(batting_rows)
bowling_df = pd.DataFrame(bowling_rows)

matches_df.to_csv("/home/claude/ipl_streamlit_app/data/matches.csv", index=False)
batting_df.to_csv("/home/claude/ipl_streamlit_app/data/batting_innings.csv", index=False)
bowling_df.to_csv("/home/claude/ipl_streamlit_app/data/bowling_innings.csv", index=False)

print(f"Matches: {len(matches_df)}")
print(f"Batting rows: {len(batting_df)}")
print(f"Bowling rows: {len(bowling_df)}")
