import os
import sys
import pandas as pd
import joblib

MODEL_PATH = "datasets/model.pkl"


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        sys.exit(f"ERROR: '{path}' not found. Run main.ipynb first to generate it.")
    bundle = joblib.load(path)
    return bundle["model"], bundle["columns"]


def predict_manual(
    model,
    cols,
    pA,
    pB,
    surface,
    series,
    round_,
    best_of,
    elo_A,
    elo_B,
    rank_A,
    rank_B,
    form_A,
    form_B,
    wr_A,
    wr_B,
):
    row = {
        "Rank_Diff": rank_A - rank_B,
        "Recent_Form_Diff": form_A - form_B,
        "Surface_WR_Diff": wr_A - wr_B,
        "Elo_Diff": elo_A - elo_B,
        "Best of": best_of,
        "Surface": surface,
        "Series": series,
        "Round": round_,
    }
    X1 = pd.get_dummies(pd.DataFrame([row]), columns=["Surface", "Series", "Round"])
    X1 = X1.reindex(columns=cols, fill_value=0)
    p = float(model.predict_proba(X1)[0, 1])

    print("\n" + "=" * 58)
    print(f"  {pA}  vs  {pB}")
    print(f"  {surface} | {series} | {round_} | Best of {best_of}")
    print("=" * 58)
    print(
        f"  {pA:<22} Elo={elo_A:>6.0f}  Rank={rank_A:>4}  Form={form_A:.2f}  {surface} WR={wr_A:.2f}"
    )
    print(
        f"  {pB:<22} Elo={elo_B:>6.0f}  Rank={rank_B:>4}  Form={form_B:.2f}  {surface} WR={wr_B:.2f}"
    )
    print("-" * 58)
    print(f"  → {pA}: {p*100:5.1f}%   |   {pB}: {(1-p)*100:5.1f}%")
    print(f"  → Predicted winner: {pA if p >= 0.5 else pB}")
    print("=" * 58 + "\n")
    return {pA: round(p, 4), pB: round(1 - p, 4)}


# ---------- input helpers ----------
def ask(prompt, cast=str, default=None, valid=None):
    """Prompt until user gives valid input. 'q' quits."""
    while True:
        hint = f" [{default}]" if default is not None else ""
        opts = f" ({'/'.join(valid)})" if valid else ""
        raw = input(f"{prompt}{opts}{hint}: ").strip()
        if raw.lower() == "q":
            sys.exit("Bye.")
        if raw == "" and default is not None:
            return default
        if valid and raw not in valid:
            print(f"  ! pick one of: {', '.join(valid)}")
            continue
        try:
            return cast(raw)
        except ValueError:
            print(f"  ! expected {cast.__name__}, got '{raw}'")


def ask_float01(prompt):
    while True:
        v = ask(prompt, cast=float)
        if 0.0 <= v <= 1.0:
            return v
        print("  ! value must be between 0 and 1 (e.g. 0.85)")


def collect_match():
    print("\n--- Match context ---")
    pA = ask("Player A name")
    pB = ask("Player B name")
    surface = ask("Surface", default="Hard", valid=["Hard", "Clay", "Grass", "Carpet"])
    series = ask(
        "Series",
        default="International",
        valid=[
            "International",
            "Grand Slam",
            "Masters 1000",
            "ATP Finals",
            "Masters",
            "ATP 500",
            "ATP 250",
        ],
    )
    round_ = ask(
        "Round",
        default="1st Round",
        valid=[
            "1st Round",
            "2nd Round",
            "3rd Round",
            "4th Round",
            "Quarterfinals",
            "Semifinals",
            "The Final",
            "Round Robin",
        ],
    )
    best_of = ask("Best of (3 or 5)", cast=int, default=3)

    print(f"\n--- {pA} pre-match stats ---")
    elo_A = ask("  Elo", cast=float)
    rank_A = ask("  ATP rank", cast=int)
    form_A = ask_float01("  Recent form (0-1)")
    wr_A = ask_float01(f"  {surface} win rate (0-1)")

    print(f"\n--- {pB} pre-match stats ---")
    elo_B = ask("  Elo", cast=float)
    rank_B = ask("  ATP rank", cast=int)
    form_B = ask_float01("  Recent form (0-1)")
    wr_B = ask_float01(f"  {surface} win rate (0-1)")

    return dict(
        pA=pA,
        pB=pB,
        surface=surface,
        series=series,
        round_=round_,
        best_of=best_of,
        elo_A=elo_A,
        elo_B=elo_B,
        rank_A=rank_A,
        rank_B=rank_B,
        form_A=form_A,
        form_B=form_B,
        wr_A=wr_A,
        wr_B=wr_B,
    )


# ---------- main loop ----------
if __name__ == "__main__":
    model, cols = load_model()
    print("Loaded model from", MODEL_PATH)
    print("Type 'q' at any prompt to quit.\n")

    while True:
        kwargs = collect_match()
        predict_manual(model, cols, **kwargs)

        again = input("Predict another match? (y/n) [n]: ").strip().lower()
        if again != "y":
            print("Done.")
            break
