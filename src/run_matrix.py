"""Orchestrator for the BERTimbau NER experiment matrix with rich progress tracking."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from runner import find_latest_checkpoint, run_experiment

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable HH:MM:SS."""
    return str(timedelta(seconds=int(round(seconds))))


def load_matrix(csv_path: Path, conditions_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load and parse experiment_matrix.csv, optionally filtering by condition names."""
    runs = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") == "neuralmind/bert-base-portuguese-cased":
                cond = row["condition"]
                if conditions_filter is not None and cond not in conditions_filter:
                    continue
                runs.append({
                    "run_id": row["run_id"],
                    "model": row["model"],
                    "condition": cond,
                    "seed": int(row["seed"]),
                    "o_class_weight": float(row["o_class_weight"]),
                    "learning_rate": float(row["learning_rate"]),
                    "batch_size": int(row["batch_size"]),
                    "primary_metric": row["primary_metric"],
                })
    return runs


def write_matrix_summary(output_root: Path, results: List[Dict[str, Any]]) -> None:
    """Write or update matrix_execution_results.csv incrementally after each run."""
    summary_csv = output_root / "matrix_execution_results.csv"
    fieldnames = [
        "run_id", "condition", "seed", "o_class_weight", "learning_rate",
        "batch_size", "gradient_accumulation_steps", "save_total_limit",
        "precision", "status", "best_metric", "best_checkpoint", "best_epoch",
        "validation_micro_f1", "test_micro_f1", "total_seconds"
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "run_id": r.get("run_id"),
                "condition": r.get("condition"),
                "seed": r.get("seed"),
                "o_class_weight": r.get("o_class_weight"),
                "learning_rate": r.get("learning_rate"),
                "batch_size": r.get("batch_size"),
                "gradient_accumulation_steps": r.get("gradient_accumulation_steps"),
                "save_total_limit": r.get("save_total_limit", 2),
                "precision": r.get("precision"),
                "status": r.get("status", "completed"),
                "best_metric": r.get("best_metric"),
                "best_checkpoint": r.get("best_checkpoint"),
                "best_epoch": r.get("best_epoch"),
                "validation_micro_f1": r.get("validation_micro_f1"),
                "test_micro_f1": r.get("test_micro_f1"),
                "total_seconds": r.get("timings", {}).get("total_seconds"),
            })


def print_matrix_table(runs: List[Dict[str, Any]]) -> None:
    """Print an ASCII table of the scheduled runs."""
    header = f"│ {'RUN_ID':<27} │ {'CONDIÇÃO':<12} │ {'SEED':<5} │ {'PESO_O':<7} │ {'LR':<8} │ {'BATCH':<5} │"
    sep_top = "┌─────────────────────────────┬──────────────┬───────┬─────────┬──────────┬───────┐"
    sep_mid = "├─────────────────────────────┼──────────────┼───────┼─────────┼──────────┼───────┤"
    sep_bot = "└─────────────────────────────┴──────────────┴───────┴─────────┴──────────┴───────┘"

    print(sep_top)
    print(header)
    print(sep_mid)
    for r in runs:
        row = (
            f"│ {r['run_id']:<27} │ {r['condition']:<12} │ {r['seed']:>5} │ "
            f"{r['o_class_weight']:>7.2f} │ {r['learning_rate']:>8.1e} │ {r['batch_size']:>5} │"
        )
        print(row)
    print(sep_bot)


def execute_matrix(
    clean_dir: Path,
    output_root: Path,
    matrix_csv: Path,
    precision: str = "fp32",
    max_epochs: int = 15,
    patience: int = 3,
    grad_accum: int = 2,
    conditions_filter: Optional[List[str]] = None,
    model_name_or_path: Optional[str] = None,
    local_files_only: bool = False,
    benchmark_mode: bool = False,
    save_total_limit: int = 2,
) -> List[Dict[str, Any]]:
    """Run BERTimbau matrix experiments sequentially with safe resumption and rich progress."""
    clean_dir = clean_dir.resolve()
    output_root = output_root.resolve()
    matrix_csv = matrix_csv.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Detect hardware environment
    is_mps = False
    if HAS_TORCH and torch is not None:
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                is_mps = True
        except Exception:
            pass

    if is_mps:
        device_desc = "Apple Silicon MPS (GPU)"
    elif HAS_TORCH and torch is not None and torch.cuda.is_available():
        device_desc = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device_desc = "CPU"

    runs = load_matrix(matrix_csv, conditions_filter=conditions_filter)

    if not runs:
        print(f"\033[1;33m[AVISO] Nenhuma experiência corresponde aos filtros fornecidos ({conditions_filter}).\033[0m")
        return []

    # 1. Informative Header before execution
    print("\n\033[1;34m======================================================================\033[0m")
    print("\033[1;34m BERTimbau NER: Execução Controlada da Matriz Experimental\033[0m")
    print("\033[1;34m======================================================================\033[0m")
    print(f" Dispositivo:           \033[1;32m{device_desc}\033[0m")
    print(f" Precisão:              \033[1;32m{precision}\033[0m")
    print(f" Total de Experiências: \033[1;36m{len(runs)}\033[0m")
    cond_list = ", ".join(sorted(set(r["condition"] for r in runs)))
    seed_list = ", ".join(str(s) for s in sorted(set(r["seed"] for r in runs)))
    print(f" Condições:             {cond_list}")
    print(f" Sementes:              {seed_list}")
    print(f" Hiperparâmetros:       Máx {max_epochs} épocas | Patience {patience} | Grad Accum {grad_accum}")
    print(f" Pasta de Saída:        {output_root}")
    print("----------------------------------------------------------------------")
    print(" Tabela das Experiências Selecionadas:")
    print_matrix_table(runs)
    print("----------------------------------------------------------------------\n")

    if benchmark_mode:
        baseline_run = next((r for r in runs if r["condition"] == "baseline" and r["seed"] == 42), runs[0])
        model_target = model_name_or_path or baseline_run["model"]
        print(f"A executar modo benchmark na run {baseline_run['run_id']}...")
        result = run_experiment(
            clean_dir=clean_dir,
            output_root=output_root,
            run_id=f"benchmark_{baseline_run['run_id']}",
            model_name_or_path=model_target,
            condition=baseline_run["condition"],
            seed=baseline_run["seed"],
            o_class_weight=baseline_run["o_class_weight"],
            learning_rate=baseline_run["learning_rate"],
            batch_size=baseline_run["batch_size"],
            max_epochs=1,
            early_stopping_patience=0,
            gradient_accumulation_steps=grad_accum,
            precision=precision,
            benchmark_mode=True,
            local_files_only=local_files_only,
            save_total_limit=save_total_limit,
        )
        return [result]

    results: List[Dict[str, Any]] = []
    matrix_start_time = time.time()
    run_durations: List[float] = []

    for i, run_info in enumerate(runs, 1):
        run_id = run_info["run_id"]
        run_dir = output_root / run_id
        summary_path = run_dir / "run_summary.json"

        # Check initial run state for transparent feedback
        initial_state = "[INÍCIO] Treino de raiz"
        if summary_path.exists():
            try:
                prev_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if prev_summary.get("status") == "completed":
                    initial_state = f"[IGNORADA] Já concluída anteriormente (Test F1={prev_summary.get('test_micro_f1', 0.0):.4f})"
            except Exception:
                pass
        if "[IGNORADA]" not in initial_state:
            latest_ckpt = find_latest_checkpoint(run_dir / "checkpoints")
            if latest_ckpt:
                initial_state = f"[RETOMA] A retomar do checkpoint: {Path(latest_ckpt).name}"

        # Timing and ETA calculation
        elapsed_total = time.time() - matrix_start_time
        if run_durations:
            avg_duration = sum(run_durations) / len(run_durations)
            remaining_runs = len(runs) - (i - 1)
            eta_str = format_duration(avg_duration * remaining_runs)
        else:
            eta_str = "A calcular após conclusão da 1ª run..."

        print("\n\033[1;35m======================================================================\033[0m")
        print(f"\033[1;35m [RUN {i}/{len(runs)}] {run_id}\033[0m ({run_info['condition']}, seed={run_info['seed']})")
        print("\033[1;35m======================================================================\033[0m")
        print(f" Hora de Início:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Tempo Decorrido:      {format_duration(elapsed_total)}")
        print(f" Estimativa Restante:  {eta_str}")
        print(f" Estado Inicial:       {initial_state}")
        print("----------------------------------------------------------------------")

        t_run_start = time.time()
        model_target = model_name_or_path or run_info["model"]
        res = run_experiment(
            clean_dir=clean_dir,
            output_root=output_root,
            run_id=run_id,
            model_name_or_path=model_target,
            condition=run_info["condition"],
            seed=run_info["seed"],
            o_class_weight=run_info["o_class_weight"],
            learning_rate=run_info["learning_rate"],
            batch_size=run_info["batch_size"],
            max_epochs=max_epochs,
            early_stopping_patience=patience,
            gradient_accumulation_steps=grad_accum,
            precision=precision,
            benchmark_mode=False,
            local_files_only=local_files_only,
            save_total_limit=save_total_limit,
        )
        t_run_end = time.time()
        run_dur = t_run_end - t_run_start
        run_durations.append(run_dur)

        results.append(res)

        # Incrementally update CSV summary table after EACH run (Requirement 2)
        write_matrix_summary(output_root, results)

        val_f1 = res.get("validation_micro_f1")
        test_f1 = res.get("test_micro_f1")
        val_f1_str = f"{val_f1:.4f}" if val_f1 is not None else "N/A"
        test_f1_str = f"{test_f1:.4f}" if test_f1 is not None else "N/A"
        best_epoch_val = res.get("best_epoch", "N/A")

        print("\n----------------------------------------------------------------------")
        print(f"\033[1;32m [FIM DA RUN {i}/{len(runs)}] {run_id}\033[0m")
        print(f"   - Estado:              {res.get('status', 'completed')}")
        print(f"   - Melhor Época:        {best_epoch_val}")
        print(f"   - Validação Micro F1:  \033[1;33m{val_f1_str}\033[0m")
        print(f"   - Teste Micro F1:      \033[1;32m{test_f1_str}\033[0m")
        print(f"   - Duração da Run:      {run_dur:.1f}s")
        print(f"   - Progresso Global:    {i}/{len(runs)} ({i / len(runs) * 100:.1f}%)")
        print(f"   - Tabela Atualizada:   {output_root / 'matrix_execution_results.csv'}")
        print("----------------------------------------------------------------------")

    total_matrix_time = time.time() - matrix_start_time
    print("\n\033[1;32m======================================================================\033[0m")
    print(f"\033[1;32m [CONCLUÍDO] Todas as {len(runs)} experiências da matriz foram finalizadas!\033[0m")
    print(f" Tempo Total:          {format_duration(total_matrix_time)}")
    print(f" Resumo Consolidado:   {output_root / 'matrix_execution_results.csv'}")
    print("\033[1;32m======================================================================\033[0m\n")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run BERTimbau NER experiments from matrix.")
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Path to the private data directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs",
        help="Directory to save experiment outputs",
    )
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "experiment_matrix.csv",
        help="Path to the experiment matrix CSV",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="Override model path or hub ID (e.g. local weights directory)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated conditions to run (e.g. 'baseline,weight_only')",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help="Compute precision (uniform across conditions)",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Force loading weights from local cache only")
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum number of checkpoints to retain")
    parser.add_argument("--benchmark", action="store_true", help="Run single-epoch baseline benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    conditions_filter = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
    execute_matrix(
        clean_dir=args.clean_dir,
        output_root=args.output_root,
        matrix_csv=args.matrix_csv,
        precision=args.precision,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grad_accum=args.grad_accum,
        conditions_filter=conditions_filter,
        model_name_or_path=args.model_name_or_path,
        local_files_only=args.local_files_only,
        benchmark_mode=args.benchmark,
        save_total_limit=args.save_total_limit,
    )


if __name__ == "__main__":
    main()
