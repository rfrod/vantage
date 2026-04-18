import sys
import os
import argparse

# Ensure the project root is on the Python path so all packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.full_graph import create_full_workflow


def load_tickers(file_path: str) -> list[str]:
    """
    Load tickers from a plain-text file.
    Supports one ticker per line, or comma/space-separated values.
    Lines starting with '#' are treated as comments and ignored.
    Empty lines are also ignored.
    """
    if not os.path.isfile(file_path):
        print(f"[ERROR] Ticker file not found: {file_path}")
        sys.exit(1)

    tickers = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Support comma-separated or space-separated tickers on the same line
            for ticker in line.replace(",", " ").split():
                t = ticker.strip().upper()
                if t:
                    tickers.append(t)

    if not tickers:
        print(f"[ERROR] No tickers found in file: {file_path}")
        sys.exit(1)

    return tickers


def parse_args():
    parser = argparse.ArgumentParser(
        prog="vantage",
        description="Vantage Investment Research System — multi-agent stock screening and decision support.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "tickers_file",
        help="Path to a plain-text file containing the ticker list.\n"
             "One ticker per line; '#' lines are comments; comma/space separation is supported."
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode.\n"
             "Prints full agent outputs, intermediate reasoning steps,\n"
             "specialist report details, debate turns, and risk assessments."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Strict mode: abort execution with a non-zero exit code\n"
             "if the Quant Screening stage finds no statistical outliers.\n"
             "Useful for automated pipelines where a no-outlier result\n"
             "should be treated as an error rather than a silent no-op."
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        default=False,
        dest="screen_only",
        help="Run only the Quant Screening stage and print the results.\n"
             "No specialist agents, debate, risk, or PM stages are executed.\n"
             "Useful for quickly identifying statistical outliers in a watchlist\n"
             "before committing to a full research run.\n"
             "Compatible with --strict (exits with code 2 if no outliers found)."
    )
    parser.add_argument(
        "--min-sigma",
        type=int,
        choices=[1, 2],
        default=1,
        dest="min_sigma",
        help="Minimum standard deviation threshold for an outlier to be analysed.\n"
             "  1 (default): process all outliers (+, ++, -, --)\n"
             "  2          : process only strong outliers (++, --) — 2-sigma events only\n"
             "Applies to both the full pipeline and --screen-only mode."
    )
    return parser.parse_args()


def print_debug(label: str, content, debug: bool):
    """Print a labelled debug block only when debug mode is active."""
    if not debug:
        return
    width = 56
    print(f"\n{'─'*width}")
    print(f"  [DEBUG] {label}")
    print(f"{'─'*width}")
    if isinstance(content, list):
        for item in content:
            print(f"  • {item}")
    elif hasattr(content, "model_dump"):
        import json
        print(json.dumps(content.model_dump(), indent=2))
    else:
        print(f"  {content}")
    print(f"{'─'*width}")


def main():
    args = parse_args()
    tickers = load_tickers(args.tickers_file)

    print(f"\n{'='*60}")
    print(f"  VANTAGE Investment Research System")
    print(f"  Loaded {len(tickers)} ticker(s) from: {args.tickers_file}")
    print(f"  Tickers     : {', '.join(tickers)}")
    print(f"  Debug       : {'ON' if args.debug else 'OFF'}")
    print(f"  Strict      : {'ON' if args.strict else 'OFF'}")
    print(f"  Screen Only : {'ON' if args.screen_only else 'OFF'}")
    print(f"  Min Sigma   : {args.min_sigma} ({'strong outliers only (++/--)' if args.min_sigma == 2 else 'all outliers (+/++/-/--)'})") 
    print(f"{'='*60}\n")

    # ── Screen-only mode ──────────────────────────────────────────────────────
    if args.screen_only:
        from agents.quant_screener import QuantScreener
        print("[Screen Only] Running Quant Screening...\n")
        screener = QuantScreener(verbose=args.debug)
        all_outliers = screener.screen_tickers(tickers)

        # Apply min_sigma filter
        if args.min_sigma >= 2:
            outliers = [o for o in all_outliers if o.classification in ("++", "--")]
            skipped = len(all_outliers) - len(outliers)
            if skipped:
                print(f"[min-sigma=2] {skipped} single-sigma outlier(s) filtered out.")
        else:
            outliers = all_outliers

        if not outliers:
            msg = "No statistical outliers detected in the provided tickers."
            if args.strict:
                print(f"[STRICT] {msg}")
                print("[STRICT] Aborting — non-zero exit code returned.\n")
                sys.exit(2)
            else:
                print(f"[INFO] {msg}\n")
                return

        print(f"{'='*60}")
        print(f"  QUANT SCREENING RESULTS — {len(outliers)} outlier(s) found")
        print(f"{'='*60}")
        for o in outliers:
            label = {
                "++": "Strong Upside Outlier",
                "+":  "Moderate Upside Outlier",
                "--": "Strong Downside Outlier",
                "-":  "Moderate Downside Outlier",
            }.get(o.classification, o.classification)
            print(f"  {o.ticker:<10}  [{o.classification}]  {label}")
        print(f"{'='*60}\n")
        return

    workflow = create_full_workflow(debug=args.debug, min_sigma=args.min_sigma)

    initial_state = {
        "tickers": tickers,
        "outliers": [],
        "current_ticker": None,
        "specialist_reports": [],
        "synthesis": None,
        "debate_history": [],
        "debate_summary": None,
        "risk_recommendations": [],
        "risk_summary": None,
        "final_decisions": [],
        "errors": []
    }

    try:
        final_state = workflow.invoke(initial_state)

        outliers = final_state.get("outliers", [])

        # ── Strict mode gate ──────────────────────────────────────────────
        if not outliers:
            msg = (
                "Quant Screening found no statistical outliers in the provided tickers.\n"
                "No agents were launched."
            )
            if args.strict:
                print(f"\n[STRICT] {msg}")
                print("[STRICT] Aborting — non-zero exit code returned.\n")
                sys.exit(2)
            else:
                print(f"\n[INFO] {msg}\n")
                return

        if args.debug:
            print_debug(
                "Quant Screening — Outliers Detected",
                [f"{o.ticker}  [{o.classification}]" for o in outliers],
                debug=True
            )

        # ── Specialist reports ────────────────────────────────────────────
        if args.debug:
            reports = final_state.get("specialist_reports", [])
            for report in reports:
                print_debug(
                    f"Specialist Report — {report.agent_name}",
                    report,
                    debug=True
                )

        # ── Synthesis ─────────────────────────────────────────────────────
        if args.debug:
            synthesis = final_state.get("synthesis")
            if synthesis:
                print_debug("Research Manager — Synthesis", synthesis, debug=True)

        # ── Debate ────────────────────────────────────────────────────────
        if args.debug:
            for turn in final_state.get("debate_history", []):
                print_debug(f"Debate Turn — {turn.agent}", turn, debug=True)
            debate_summary = final_state.get("debate_summary")
            if debate_summary:
                print_debug("Debate Moderator — Summary", debate_summary, debug=True)

        # ── Risk ──────────────────────────────────────────────────────────
        if args.debug:
            for rec in final_state.get("risk_recommendations", []):
                print_debug(f"Risk Analyst — {rec.persona}", rec, debug=True)
            risk_summary = final_state.get("risk_summary")
            if risk_summary:
                print_debug("Risk Committee Chair — Summary", risk_summary, debug=True)

        # ── Final decisions ───────────────────────────────────────────────
        decisions = final_state.get("final_decisions", [])
        if not decisions:
            print("[INFO] No final decisions produced.")
        else:
            for d in decisions:
                print(f"\n{'='*60}")
                print(f"  FINAL DECISION — {d.ticker}")
                print(f"{'='*60}")
                print(f"  Action:          {d.final_action}")
                print(f"  Confidence:      {d.confidence_score}/10")
                print(f"  Position Size:   {d.recommended_position_size}")
                if d.recommended_strike_expiration and d.recommended_strike_expiration != "N/A":
                    print(f"  Strike/Exp:      {d.recommended_strike_expiration}")
                if d.spread_details:
                    print(f"  Spread Details:  {d.spread_details}")
                print(f"\n  Top Reasons:")
                for i, r in enumerate(d.top_3_reasons, 1):
                    print(f"    {i}. {r}")
                print(f"\n  Top Risks:")
                for i, r in enumerate(d.top_3_risks, 1):
                    print(f"    {i}. {r}")
                print(f"\n  Monitor Next:")
                for m in d.what_to_monitor_next:
                    print(f"    - {m}")
                if args.debug:
                    print(f"\n  Audit Trail:")
                    for a in d.audit_trail:
                        print(f"    • {a}")
                print(f"\n  Report saved to: outputs/{d.ticker}_report.json")
                print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n[ERROR] Execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
