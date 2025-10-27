import pandas as pd
import sys


def analyze_report(csv_path: str):
    """Analyze DER report and provide insights."""
    df = pd.read_csv(csv_path)

    # Filter successful runs
    success = df[df["status"] == "success"]

    if len(success) == 0:
        print("No successful recordings!")
        return

    print("\n" + "=" * 70)
    print("DIARIZATION ERROR RATE ANALYSIS")
    print("=" * 70)

    # Overall stats
    print(f"\nTotal recordings: {len(success)}")
    print(
        f"Average DER: {success['DER'].mean():.4f} ({success['DER'].mean()*100:.2f}%)"
    )
    print(f"Median DER: {success['DER'].median():.4f}")
    print(f"Best DER: {success['DER'].min():.4f}")
    print(f"Worst DER: {success['DER'].max():.4f}")

    # Error component breakdown
    print(f"\n{'='*70}")
    print("ERROR COMPONENTS (Avg)")
    print(f"{'='*70}")
    print(
        f"False Alarm:       {success['false_alarm'].mean():.4f} ({success['false_alarm'].mean()*100:.2f}%)"
    )
    print(
        f"Missed Detection:  {success['missed_detection'].mean():.4f} ({success['missed_detection'].mean()*100:.2f}%)"
    )
    print(
        f"Confusion:         {success['confusion'].mean():.4f} ({success['confusion'].mean()*100:.2f}%)"
    )

    # Speech detection analysis
    print(f"\n{'='*70}")
    print("SPEECH DETECTION")
    print(f"{'='*70}")
    print(f"Avg missing speech: {success['missing_speech_pct'].mean():.2f}%")
    over_detection = success[success["missing_speech_pct"] < 0]
    print(f"Over-detecting speech: {len(over_detection)}/{len(success)} recordings")
    print(
        f"Under-detecting speech: {len(success) - len(over_detection)}/{len(success)} recordings"
    )

    # Categorize problems
    print(f"\n{'='*70}")
    print("WORST PERFORMERS (DER > 0.5)")
    print(f"{'='*70}")
    worst = success[success["DER"] > 0.5].sort_values("DER", ascending=False)
    for _, row in worst.iterrows():
        print(f"\n{row['rec_id']}: DER={row['DER']:.4f}")
        print(
            f"  FA={row['false_alarm']:.3f}, MD={row['missed_detection']:.3f}, Conf={row['confusion']:.3f}"
        )
        print(f"  Missing speech: {row['missing_speech_pct']:.1f}%")

    # Best performers
    print(f"\n{'='*70}")
    print("BEST PERFORMERS (DER < 0.2)")
    print(f"{'='*70}")
    best = success[success["DER"] < 0.2].sort_values("DER")
    for _, row in best.iterrows():
        print(f"{row['rec_id']}: DER={row['DER']:.4f}")

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")

    avg_fa = success["false_alarm"].mean()
    avg_md = success["missed_detection"].mean()
    avg_conf = success["confusion"].mean()

    if avg_fa > 0.15:
        print("\n🔴 HIGH FALSE ALARM (Detecting non-speech as speech)")
        print("   → Increase VAD threshold (make it LESS sensitive)")
        print("   → Add silence filtering")
        print("   → Check for background noise in audio")

    if avg_md > 0.15:
        print("\n🔴 HIGH MISSED DETECTION (Missing actual speech)")
        print("   → Decrease VAD threshold (make it MORE sensitive)")
        print("   → Check audio quality/volume levels")

    if avg_conf > 0.20:
        print("\n🔴 HIGH CONFUSION (Wrong speaker labels)")
        print("   → Speaker embeddings not discriminative enough")
        print("   → May need speaker-specific fine-tuning")
        print("   → Check if speakers have similar voices")

    if success["missing_speech_pct"].mean() < -5:
        print("\n🔴 OVER-DETECTION (System detecting too much speech)")
        print("   → Strongly increase VAD threshold")
        print("   → System is treating silence/noise as speech")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <report.csv>")
        sys.exit(1)

    analyze_report(sys.argv[1])
