from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


RUNNING = True


# Map snake_case CICFlowMeter-style realtime columns to the original
# CICFlowMeter headers used by the training datasets.
CICFLOWMETER_COLUMN_ALIASES = {
    "flow_id": "Flow ID",
    "src_ip": "Src IP",
    "dst_ip": "Dst IP",
    "src_port": "Src Port",
    "dst_port": "Dst Port",
    "protocol": "Protocol",
    "timestamp": "Timestamp",
    "flow_duration": "Flow Duration",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "tot_fwd_pkts": "Total Fwd Packet",
    "tot_bwd_pkts": "Total Bwd packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packet",
    "totlen_bwd_pkts": "Total Length of Bwd Packet",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "pkt_len_max": "Packet Length Max",
    "pkt_len_min": "Packet Length Min",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_seg_size_min": "Fwd Seg Size Min",
    "fwd_act_data_pkts": "Fwd Act Data Pkts",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "flow_iat_std": "Flow IAT Std",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "ece_flag_cnt": "ECE Flag Count",
    "cwr_flag_count": "CWR Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Average Packet Size",
    "init_fwd_win_byts": "FWD Init Win Bytes",
    "init_bwd_win_byts": "Bwd Init Win Bytes",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "fwd_byts_b_avg": "Fwd Bytes/Bulk Avg",
    "fwd_pkts_b_avg": "Fwd Packet/Bulk Avg",
    "bwd_byts_b_avg": "Bwd Bytes/Bulk Avg",
    "bwd_pkts_b_avg": "Bwd Packet/Bulk Avg",
    "fwd_blk_rate_avg": "Fwd Bulk Rate Avg",
    "bwd_blk_rate_avg": "Bwd Bulk Rate Avg",
    "fwd_seg_size_avg": "Fwd Segment Size Avg",
    "bwd_seg_size_avg": "Bwd Segment Size Avg",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
    "label": "Label",
    "attack_name": "Attack Name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime IDS inference from CICFlowMeter CSV."
    )
    parser.add_argument(
        "--model-bundle",
        type=Path,
        required=True,
        help="Path to best_model.joblib or equivalent bundle.",
    )
    parser.add_argument(
        "--flows-csv",
        type=Path,
        required=True,
        help="Path to live CSV appended by CICFlowMeter.",
    )
    parser.add_argument(
        "--alerts-csv",
        type=Path,
        default=Path("realtime_alerts.csv"),
        help="CSV file to append scored predictions.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("realtime_state.json"),
        help="Stores processed row count for resume.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Ignore existing state file and start state tracking from scratch for this run.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=None,
        help="Override saved threshold from bundle.",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Start from end of existing CSV and process only new rows.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print benign and attack rows. Default prints only attacks.",
    )
    parser.add_argument(
        "--log-all",
        action="store_true",
        help="Write all predictions to alerts CSV. Default writes only attacks.",
    )
    parser.add_argument(
        "--src-ip-column",
        type=str,
        default="src_ip",
        help="Source IP column name.",
    )
    parser.add_argument(
        "--dst-ip-column",
        type=str,
        default="dst_ip",
        help="Destination IP column name.",
    )
    parser.add_argument(
        "--src-port-column",
        type=str,
        default="src_port",
        help="Source port column name.",
    )
    parser.add_argument(
        "--dst-port-column",
        type=str,
        default="dst_port",
        help="Destination port column name.",
    )
    parser.add_argument(
        "--protocol-column",
        type=str,
        default="protocol",
        help="Protocol column name.",
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default="timestamp",
        help="Timestamp column name.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Label column to drop if present.",
    )
    parser.add_argument(
        "--no-aggregate-attacks",
        action="store_true",
        help="Disable grouped attack alerts and print/log every attack flow separately.",
    )
    return parser.parse_args()


def install_signal_handlers() -> None:
    def _handle_signal(signum: int, frame: Any) -> None:
        del signum, frame
        global RUNNING
        RUNNING = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def load_bundle(bundle_path: Path) -> tuple[Any, float, dict[str, Any]]:
    bundle = joblib.load(bundle_path)

    if isinstance(bundle, dict):
        if "pipeline" not in bundle:
            raise KeyError("Bundle dict must contain 'pipeline'.")
        pipeline = bundle["pipeline"]
        threshold = float(bundle.get("threshold", 0.5))
        metadata = bundle.get("metadata", {})
        return pipeline, threshold, metadata

    threshold = 0.5
    metadata: dict[str, Any] = {}
    return bundle, threshold, metadata


def get_expected_columns(pipeline: Any, metadata: dict[str, Any]) -> list[str]:
    metadata_cols = metadata.get("feature_columns")
    if isinstance(metadata_cols, list) and metadata_cols:
        return list(metadata_cols)

    if not hasattr(pipeline, "named_steps"):
        raise KeyError(
            "Cannot infer feature columns: pipeline has no named_steps and metadata.feature_columns is missing."
        )

    if "preprocessor" not in pipeline.named_steps:
        raise KeyError(
            "Cannot infer feature columns: pipeline has no 'preprocessor' step and metadata.feature_columns is missing."
        )

    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(preprocessor, "transformers_"):
        raise KeyError(
            "Cannot infer feature columns: preprocessor has no transformers_."
        )

    columns: list[str] = []
    for _, _, transformer_columns in preprocessor.transformers_:
        if transformer_columns is None:
            continue
        if isinstance(transformer_columns, list):
            columns.extend(transformer_columns)
        else:
            try:
                columns.extend(list(transformer_columns))
            except TypeError:
                columns.append(transformer_columns)

    deduped: list[str] = []
    seen = set()
    for col in columns:
        if col not in seen:
            seen.add(col)
            deduped.append(col)

    if not deduped:
        raise ValueError("Expected feature column list is empty.")

    return deduped


def build_file_signature(path: Path) -> str:
    if not path.exists():
        return ""

    try:
        stat = path.stat()
    except OSError:
        return ""

    return ":".join(
        [
            str(getattr(stat, "st_dev", "")),
            str(getattr(stat, "st_ino", "")),
            str(getattr(stat, "st_ctime_ns", "")),
        ]
    )


def load_state(
    state_file: Path,
    tail: bool,
    flows_csv: Path,
    reset_state: bool = False,
) -> dict[str, Any]:
    if reset_state and state_file.exists():
        try:
            state_file.unlink()
        except Exception:
            pass

    if state_file.exists():
        try:
            with state_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            state = {
                "processed_rows": int(data.get("processed_rows", 0)),
                "file_signature": str(data.get("file_signature", "")),
            }

            current_signature = build_file_signature(flows_csv)
            if current_signature and state["file_signature"] and current_signature != state["file_signature"]:
                return {"processed_rows": 0, "file_signature": current_signature}

            if current_signature and not state["file_signature"]:
                state["file_signature"] = current_signature
            return state
        except Exception:
            pass

    state: dict[str, Any] = {
        "processed_rows": 0,
        "file_signature": build_file_signature(flows_csv),
    }
    if tail and flows_csv.exists():
        try:
            with flows_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                state["processed_rows"] = max(sum(1 for _ in handle) - 1, 0)
        except Exception:
            state["processed_rows"] = 0
    return state


def save_state(state_file: Path, state: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def ensure_alert_header(alerts_csv: Path) -> None:
    alerts_csv.parent.mkdir(parents=True, exist_ok=True)
    if alerts_csv.exists() and alerts_csv.stat().st_size > 0:
        return

    with alerts_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "logged_at",
                "flow_timestamp",
                "prediction",
                "score",
                "threshold",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "protocol",
                "model_name",
                "sampling_strategy",
                "flow_count",
                "unique_src_ports",
                "unique_dst_ports",
                "max_score",
            ]
        )


def append_alert(alerts_csv: Path, row: list[Any]) -> None:
    with alerts_csv.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


def safe_read_csv_skip_partial(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None

    if not text.strip():
        return None

    lines = text.splitlines(keepends=True)
    if len(lines) <= 1:
        return None

    if not lines[-1].endswith("\n"):
        lines = lines[:-1]

    if len(lines) <= 1:
        return None

    try:
        return pd.read_csv(StringIO("".join(lines)), low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def normalize_label_column_name(df: pd.DataFrame, requested_label: str) -> str:
    if requested_label in df.columns:
        return requested_label

    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(requested_label.lower(), requested_label)


def rename_realtime_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        mapped = CICFLOWMETER_COLUMN_ALIASES.get(col.lower())
        if mapped is not None and mapped not in df.columns:
            rename_map[col] = mapped
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def prepare_features(
    rows_df: pd.DataFrame,
    expected_columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    working = rename_realtime_columns(rows_df.copy())

    label_column = normalize_label_column_name(working, label_column)
    if label_column in working.columns:
        working = working.drop(columns=[label_column])

    for column in expected_columns:
        if column not in working.columns:
            working[column] = pd.NA

    working = working[expected_columns].copy()

    for col in expected_columns:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    working = working.replace([float("inf"), float("-inf")], pd.NA)

    return working


def summarize_schema(
    rows_df: pd.DataFrame,
    expected_columns: list[str],
    label_column: str,
) -> dict[str, Any]:
    working = rename_realtime_columns(rows_df.copy())
    label_column = normalize_label_column_name(working, label_column)

    if label_column in working.columns:
        working = working.drop(columns=[label_column])

    actual_columns = list(working.columns)
    actual_set = set(actual_columns)
    expected_set = set(expected_columns)

    matched = [col for col in expected_columns if col in actual_set]
    missing = [col for col in expected_columns if col not in actual_set]
    extra = [col for col in actual_columns if col not in expected_set]

    return {
        "matched_count": len(matched),
        "expected_count": len(expected_columns),
        "missing": missing,
        "extra": extra,
    }


def score_rows(
    pipeline: Any,
    threshold: float,
    rows_df: pd.DataFrame,
    expected_columns: list[str],
    label_column: str,
) -> tuple[pd.DataFrame, list[float], list[int]]:
    X = prepare_features(rows_df, expected_columns, label_column)

    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        raw_scores = pd.Series(pipeline.decision_function(X), dtype="float64")
        if raw_scores.max() > raw_scores.min():
            scores = ((raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())).to_numpy()
        else:
            scores = raw_scores.to_numpy() * 0.0
    else:
        preds = pipeline.predict(X)
        scores = pd.Series(preds, dtype="float64").to_numpy()

    predictions = (scores >= threshold).astype(int)
    return X, scores.tolist(), predictions.tolist()


def print_prediction(
    row: pd.Series,
    prediction: int,
    score: float,
    threshold: float,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if prediction == 0 and not args.print_all:
        return

    label = "ATTACK" if prediction == 1 else "BENIGN"
    flow_ts = row.get(args.timestamp_column, "-")
    src_ip = row.get(args.src_ip_column, "-")
    src_port = row.get(args.src_port_column, "-")
    dst_ip = row.get(args.dst_ip_column, "-")
    dst_port = row.get(args.dst_port_column, "-")
    proto = row.get(args.protocol_column, "-")
    model_name = metadata.get("model_name", "unknown")
    strategy = metadata.get("sampling_strategy", "unknown")

    print(
        f"[{label}] flow_ts={flow_ts} "
        f"{src_ip}:{src_port} -> {dst_ip}:{dst_port} proto={proto} "
        f"score={score:.4f} threshold={threshold:.4f} "
        f"model={model_name} strategy={strategy}",
        flush=True,
    )


def should_log(prediction: int, args: argparse.Namespace) -> bool:
    return args.log_all or prediction == 1


def aggregate_attack_rows(
    rows_df: pd.DataFrame,
    scores: list[float],
    predictions: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    attack_indices = [idx for idx, pred in enumerate(predictions) if int(pred) == 1]
    if not attack_indices:
        return []

    grouped: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for idx in attack_indices:
        row = rows_df.iloc[idx]
        src_ip = row.get(args.src_ip_column, "")
        dst_ip = row.get(args.dst_ip_column, "")
        proto = row.get(args.protocol_column, "")
        key = (src_ip, dst_ip, proto)

        if key not in grouped:
            grouped[key] = {
                "row": row,
                "scores": [],
                "src_ports": set(),
                "dst_ports": set(),
                "flow_count": 0,
            }

        grouped[key]["scores"].append(float(scores[idx]))
        grouped[key]["flow_count"] += 1
        grouped[key]["src_ports"].add(str(row.get(args.src_port_column, "")))
        grouped[key]["dst_ports"].add(str(row.get(args.dst_port_column, "")))

    summaries: list[dict[str, Any]] = []
    for group in grouped.values():
        score_list = sorted(group["scores"], reverse=True)
        summaries.append(
            {
                "row": group["row"],
                "flow_count": group["flow_count"],
                "max_score": score_list[0],
                "src_ports": sorted(p for p in group["src_ports"] if p),
                "dst_ports": sorted(p for p in group["dst_ports"] if p),
            }
        )
    return summaries


def process_new_rows(
    args: argparse.Namespace,
    pipeline: Any,
    threshold: float,
    metadata: dict[str, Any],
    state: dict[str, Any],
    expected_columns: list[str],
) -> int:
    df = safe_read_csv_skip_partial(args.flows_csv)
    if df is None:
        return 0

    current_signature = build_file_signature(args.flows_csv)
    previous_signature = str(state.get("file_signature", ""))
    if current_signature and previous_signature and current_signature != previous_signature:
        state["processed_rows"] = 0
    state["file_signature"] = current_signature

    total_rows = len(df)
    processed_rows = int(state.get("processed_rows", 0))

    if total_rows < processed_rows:
        processed_rows = 0
        state["processed_rows"] = 0

    if total_rows <= processed_rows:
        return 0

    new_df = df.iloc[processed_rows:].copy().reset_index(drop=True)

    _, scores, predictions = score_rows(
        pipeline=pipeline,
        threshold=threshold,
        rows_df=new_df,
        expected_columns=expected_columns,
        label_column=args.label_column,
    )

    ensure_alert_header(args.alerts_csv)
    logged_at = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.no_aggregate_attacks:
        for idx, (_, row) in enumerate(new_df.iterrows()):
            score = float(scores[idx])
            prediction = int(predictions[idx])

            print_prediction(row, prediction, score, threshold, metadata, args)

            if should_log(prediction, args):
                append_alert(
                    args.alerts_csv,
                    [
                        logged_at,
                        row.get(args.timestamp_column, ""),
                        prediction,
                        score,
                        threshold,
                        row.get(args.src_ip_column, ""),
                        row.get(args.src_port_column, ""),
                        row.get(args.dst_ip_column, ""),
                        row.get(args.dst_port_column, ""),
                        row.get(args.protocol_column, ""),
                        metadata.get("model_name", ""),
                        metadata.get("sampling_strategy", ""),
                        1,
                        row.get(args.src_port_column, ""),
                        row.get(args.dst_port_column, ""),
                        score,
                    ],
                )
    else:
        if args.print_all:
            benign_count = sum(1 for pred in predictions if int(pred) == 0)
            if benign_count:
                print(f"[INFO] benign_flows_in_batch={benign_count}", flush=True)

        summaries = aggregate_attack_rows(new_df, scores, predictions, args)
        for item in summaries:
            row = item["row"]
            max_score = float(item["max_score"])
            flow_count = int(item["flow_count"])
            src_ports = ",".join(item["src_ports"][:8])
            dst_ports = ",".join(item["dst_ports"][:12])
            flow_ts = row.get(args.timestamp_column, "-")
            src_ip = row.get(args.src_ip_column, "-")
            dst_ip = row.get(args.dst_ip_column, "-")
            proto = row.get(args.protocol_column, "-")
            model_name = metadata.get("model_name", "unknown")
            strategy = metadata.get("sampling_strategy", "unknown")

            print(
                f"[ATTACK] flow_ts={flow_ts} src={src_ip} dst={dst_ip} proto={proto} "
                f"flows={flow_count} max_score={max_score:.4f} threshold={threshold:.4f} "
                f"dst_ports={dst_ports} model={model_name} strategy={strategy}",
                flush=True,
            )

            append_alert(
                args.alerts_csv,
                [
                    logged_at,
                    flow_ts,
                    1,
                    max_score,
                    threshold,
                    src_ip,
                    src_ports,
                    dst_ip,
                    dst_ports,
                    proto,
                    model_name,
                    strategy,
                    flow_count,
                    src_ports,
                    dst_ports,
                    max_score,
                ],
            )

    state["processed_rows"] = total_rows
    save_state(args.state_file, state)
    return len(new_df)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    install_signal_handlers()

    pipeline, saved_threshold, metadata = load_bundle(args.model_bundle)
    threshold = float(args.alert_threshold) if args.alert_threshold is not None else saved_threshold
    expected_columns = get_expected_columns(pipeline, metadata)
    state = load_state(
        args.state_file,
        args.tail,
        args.flows_csv,
        reset_state=args.reset_state,
    )

    print(f"Model bundle: {args.model_bundle}")
    print(f"Flows CSV: {args.flows_csv}")
    print(f"Alerts CSV: {args.alerts_csv}")
    print(f"State file: {args.state_file}")
    print(f"Reset state: {args.reset_state}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Expected raw feature columns: {len(expected_columns)}")
    print(f"Model: {metadata.get('model_name', 'unknown')}")
    print(f"Strategy: {metadata.get('sampling_strategy', 'unknown')}")
    if threshold <= 0:
        print(
            "[WARN] Threshold <= 0.0. This will classify almost everything as ATTACK. "
            "Check the bundle or override with --alert-threshold.",
            flush=True,
        )

    initial_df = safe_read_csv_skip_partial(args.flows_csv)
    if initial_df is not None and not initial_df.empty:
        schema_info = summarize_schema(
            initial_df,
            expected_columns=expected_columns,
            label_column=args.label_column,
        )
        print(
            f"Schema match: {schema_info['matched_count']}/{schema_info['expected_count']} expected columns",
            flush=True,
        )
        if schema_info["missing"]:
            preview = ", ".join(schema_info["missing"][:10])
            suffix = " ..." if len(schema_info["missing"]) > 10 else ""
            print(f"Missing expected columns: {preview}{suffix}", flush=True)
        if schema_info["extra"]:
            preview = ", ".join(schema_info["extra"][:10])
            suffix = " ..." if len(schema_info["extra"]) > 10 else ""
            print(f"Extra realtime columns: {preview}{suffix}", flush=True)
    print("Starting realtime inference loop...", flush=True)

    processed_total = 0
    while RUNNING:
        processed_now = process_new_rows(
            args=args,
            pipeline=pipeline,
            threshold=threshold,
            metadata=metadata,
            state=state,
            expected_columns=expected_columns,
        )
        processed_total += processed_now
        if processed_now == 0:
            time.sleep(args.poll_interval)

    print(
        f"Stopping realtime inference. Total processed new rows: {processed_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()
