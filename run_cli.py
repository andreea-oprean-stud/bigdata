"""
Command Line Interface (CLI) for Satellite Pass Prediction & Big Data Platform.
Provides terminal-based execution for pass predictions, data lake generation, and benchmarks.
"""

import argparse
import sys
from datetime import datetime, timezone
from src.data_ingestion import CelesTrakClient, N2YOClient, SAMPLE_TLES
from src.orbital_engine import SGP4Engine, PassPredictor
from src.big_data_pipeline import HistoricalOrbitalGenerator, ScalabilityBenchmark, SparkOrbitalAnalytics, ParquetLakeManager
from config import OBSERVER_PRESETS, POPULAR_SATELLITES


def main():
    parser = argparse.ArgumentParser(
        description="Big Data Platform for Satellite Pass Prediction & Orbital Analytics"
    )
    parser.add_argument("--predict", action="store_true", help="Run pass prediction")
    parser.add_argument("--satellite", type=int, default=25544, help="Target NORAD Catalog ID (default: 25544 ISS)")
    parser.add_argument("--observer", type=str, default="Bucharest, Romania", help="Observer city preset")
    parser.add_argument("--days", type=int, default=3, help="Prediction window in days")
    parser.add_argument("--min-elevation", type=float, default=10.0, help="Minimum elevation threshold in degrees")
    parser.add_argument("--generate", type=int, metavar="N", help="Generate N orbital records into Spark Parquet Lake")
    parser.add_argument("--benchmark", action="store_true", help="Run scalability and storage compression benchmarks")
    parser.add_argument("--stats", action="store_true", help="Print Parquet Lake statistics")

    args = parser.parse_args()

    if args.predict:
        obs = OBSERVER_PRESETS.get(args.observer, OBSERVER_PRESETS["Bucharest, Romania"])
        print(f"\n=======================================================")
        print(f"🛰️ PREDICTING PASSES FOR NORAD {args.satellite}")
        print(f"📍 OBSERVER: {args.observer} ({obs['lat']}°N, {obs['lon']}°E, Alt: {obs['alt']}m)")
        print(f"=======================================================\n")

        client = CelesTrakClient()
        tle = client.get_tle_by_catnr(args.satellite)
        if not tle:
            tle = SAMPLE_TLES.get(args.satellite, SAMPLE_TLES[25544])

        engine = SGP4Engine(tle["line1"], tle["line2"], name=tle.get("name", "SAT"))
        predictor = PassPredictor(engine)

        passes = predictor.predict_passes(
            obs["lat"], obs["lon"], obs["alt"], duration_days=args.days, min_elevation_deg=args.min_elevation
        )

        print(f"Found {len(passes)} passes above {args.min_elevation}° in the next {args.days} days:\n")
        for p in passes:
            print(f"• Pass #{p['pass_id']}: {p['aos_time_utc']} UTC -> Max El: {p['max_elevation_deg']}° at {p['tca_time_utc']} ({p['visibility_type']})")
        print()

    elif args.generate:
        print(f"\n🚀 Generating {args.generate:,} records into Parquet Lake via PySpark...")
        generator = HistoricalOrbitalGenerator()
        summary = generator.generate_and_save_dataset(total_records=args.generate)
        print(f"✅ Generation complete in {summary['total_time_sec']}s (Throughput: {summary['throughput_records_per_sec']:,} records/sec)")
        print(f"📁 Parquet Lake Size: {summary['lake_size_mb']} MB across {summary['parquet_files']} files.\n")

    elif args.benchmark:
        print("\n🧪 Running Scalability & Storage Compression Benchmarks...")
        bench = ScalabilityBenchmark()
        print("\n1. Storage Footprint Comparison (CSV vs Parquet Snappy vs Parquet Gzip):")
        res = bench.run_storage_efficiency_test(record_count=50000)
        print(f"   • Raw CSV: {res['csv_size_mb']} MB")
        print(f"   • Parquet Snappy: {res['parquet_snappy_size_mb']} MB (Compression: {res['snappy_compression_ratio']}x, Savings: {res['storage_savings_pct']}%)")
        print(f"   • Parquet Gzip: {res['parquet_gzip_size_mb']} MB (Compression: {res['gzip_compression_ratio']}x)")

        print("\n2. Spark Scaling Benchmark (20k -> 100k records):")
        scale_df = bench.run_scaling_benchmark([20000, 50000, 100000])
        print(scale_df.to_string(index=False))
        print()

    elif args.stats:
        lake = ParquetLakeManager()
        stats = lake.get_lake_stats()
        print("\n================ PARQUET LAKE STATS ================")
        print(f"• Total Records: {stats['total_records']:,}")
        print(f"• Total Size: {stats['total_size_mb']} MB")
        print(f"• File Count: {stats['file_count']}")
        print(f"• Partitions Discovered: {len(stats['partitions'])}")
        print("====================================================\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
