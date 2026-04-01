import os
import sys
import json
import argparse
import datetime


def atomic_write_json(final_path, payload):
    """Atomically write JSON to final_path using a .tmp file and os.replace."""
    tmp_path = final_path + ".tmp"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        return True, None
    except Exception as e:
        # fallback non-atomic write to avoid leaving tmp files
        try:
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            return True, f"fallback-non-atomic: {e}"
        except Exception as e2:
            return False, str(e2)


REQUIRED_KEYS = [
    'model_name', 'run_id', 'timestamp',
    'cv_r2_mean', 'cv_rmse_mean', 'cv_mae_mean',
    'holdout_r2', 'holdout_rmse', 'holdout_mae',
    'selected_k', 'use_synthetic', 'synthetic_only',
    'augment_mode', 'augment_file', 'augment_size', 'augment_size_effective', 'augment_seed',
    'fallback_percent', 'sigma_resid_factor', 'features_count'
]


def load_aggregated(metrics_dir):
    path = os.path.join(metrics_dir, 'all_models_metrics.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def validate_schema(data, strict=False):
    missing = [k for k in REQUIRED_KEYS if k not in data]
    notes = []
    if missing:
        notes.append(f"missing_keys={missing}")
    # lightweight type checks if strict
    if strict:
        try:
            float(data.get('cv_r2_mean'))
            float(data.get('cv_rmse_mean'))
            float(data.get('cv_mae_mean'))
        except Exception:
            notes.append('type_error: cv metrics not numeric')
    return notes


def audit_and_backfill(dataset, metrics_dir, run_id=None, backfill=False, strict=False):
    agg = load_aggregated(metrics_dir)
    rows = []
    missing_count = 0
    backfilled_count = 0

    # Determine target models to audit
    target = {}
    for model_name, payload in agg.items():
        if not isinstance(payload, dict):
            continue
        if run_id is None or payload.get('run_id') == run_id:
            target[model_name] = payload

    if not target:
        print(f"No aggregated entries found for dataset={dataset} run_id={run_id}.")
        return 1

    for model_name, payload in target.items():
        final_name = f"{model_name}_metrics_{payload.get('run_id')}.json"
        final_path = os.path.join(metrics_dir, final_name)
        exists = os.path.exists(final_path)
        notes = []
        backfilled = False

        if exists:
            try:
                with open(final_path, 'r', encoding='utf-8') as f:
                    per_model = json.load(f)
                notes.extend(validate_schema(per_model, strict=strict))
            except Exception as e:
                notes.append(f"read_error: {e}")
        else:
            missing_count += 1
            notes.append('missing_per_model_json')
            if backfill:
                # Ensure minimal metadata present
                payload = dict(payload)
                payload.setdefault('timestamp', datetime.datetime.now().isoformat())
                ok, err = atomic_write_json(final_path, payload)
                if ok:
                    backfilled = True
                    backfilled_count += 1
                    if err:
                        notes.append(err)
                else:
                    notes.append(f"backfill_error: {err}")

        rows.append((model_name, exists, backfilled, '; '.join(notes) if notes else ''))

    # Print table
    print("model | per_model_json_exists | backfilled | notes")
    print("----- | --------------------- | ---------- | -----")
    for m, ex, bf, nt in rows:
        print(f"{m} | {str(ex)} | {str(bf)} | {nt}")

    if missing_count > 0 and not backfill:
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description='Audit and backfill per-model metrics JSONs')
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--metrics_dir', type=str, required=True)
    ap.add_argument('--run_id', type=str, default=None)
    ap.add_argument('--backfill', action='store_true')
    ap.add_argument('--strict', action='store_true')
    args = ap.parse_args()

    exit_code = audit_and_backfill(args.dataset, args.metrics_dir, run_id=args.run_id, backfill=args.backfill, strict=args.strict)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()