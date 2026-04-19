"""
runner.py — orchestrates the full MageZero training pipeline.

For each generation:
  1. Resolve curriculum settings
  2. For each opponent (sequential):
       reserve session ids for primary + opponent
       build a mutated game.yml
       start servers (skipped for offline)
       launch JVM, write hdf5 files, stop servers
  3. Optional: dataset_stats on new data
  4. Optional: eval previous model on new data
  5. Move testing/ → training/
  6. Archive out-of-window training files
  7. Train (50 epochs bootstrap, 10 epochs from checkpoint)
  8. Restore archived files
  9. Record gen completion in the run file
"""
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional
from itertools import combinations
import yaml
from dataclasses import asdict

from magezero.util.config import (
    CurriculumConfig,
    GenSettings,
    Opponent,
    RunConfig,
    RunRoundConfig,
    load_all,
    resolve_gen,
)
from .runner import find_active_run, update_run, copy_starting_checkpoint, create_run_dir,\
    has_checkpoint, latest_version, next_session_id, primary_file, opponent_file, build_game_yml,\
    start_server, stop_server, launch_jvm, run_dataset_stats, run_test, move_testing_to_training, \
    archive_out_of_window, restore_from_archive, run_train, record_gen

# ─── constants ───────────────────────────────────────────────

PRIMARY_PORT = 50052
OPPONENT_PORT = 50053
TMP_GAME_YML = ".mz_tmp/game.yml"
RUNS_DIR = Path("runs")
SRC = "src/magezero"
PYTHON = sys.executable
EPOCHS_BOOTSTRAP = 50
EPOCHS_ONLINE = 10
MAGE_DIR = "/home/raven/Fendley/MagicAI/mage"


# ─── game.yml mutation ───────────────────────────────────────

def build_round_game_yml(base_path: str, settings: GenSettings, run: RunConfig,
                   player: Opponent, opp: Opponent, primary_out: Path, opponent_out: Path,
                   primary_offline: bool, opp_offline: bool) -> str:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    decks_dir = Path(f"{MAGE_DIR}/decks").resolve()

    pa = cfg["player_a"]
    pa["deckPath"] = str(decks_dir / f"{player.deck}.dck")
    pa["output_file"] = str(primary_out.resolve())
    pa["mcts"]["offline_mode"] = primary_offline
    pa["mcts"]["td_discount"] = settings.td_discount
    pa["priors"]["prior_temperature"] = settings.prior_temperature
    pa["priors"]["binary"] = settings.priors.binary
    pa["priors"]["priority"] = settings.priors.priority
    pa["priors"]["target"] = settings.priors.target
    pa["priors"]["opponent"] = settings.priors.opponent

    pb = cfg["player_b"]
    pb["deckPath"] = str(decks_dir / f"{opp.deck}.dck")
    pb["output_file"] = str(opponent_out.resolve())
    pb["type"] = "minimax" if opp.mode == "minimax" else "mcts"
    pb["mcts"]["offline_mode"] = opp_offline

    cfg["training"]["games"] = run.games_per_gen
    cfg["server"]["port"] = PRIMARY_PORT
    cfg["server"]["opponent_port"] = OPPONENT_PORT

    out = Path(TMP_GAME_YML)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(out)

def create_round_dir(run: RunRoundConfig) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir()
    data = {
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "abandoned_at": None,
        # "primary": {"deck": run.deck, "version": run.version},
        # "opponents": [
        #     {"deck": o.deck, "mode": o.mode, "version": o.version}
        #     for o in run.opponents
        # ],
        "deck_pool": [
            {"deck": o.deck, "mode": o.mode, "version": o.version}
            for o in run.deck_pool
        ],
        "run_config_snapshot": {
            "generations": run.generations,
            "games_per_gen": run.games_per_gen,
            "replay_buffer_gens": run.replay_buffer_gens,
            "start_from_version": run.start_from_version,
        },
        "current_gen": 0,
        "stage": "init",
        "gens": {},
    }
    (run_dir / "run.json").write_text(json.dumps(data, indent=2))
    return run_dir

def find_active_round_run(deck_pool: list[Opponent], version: int) -> Optional[Path]:
    if not RUNS_DIR.exists():
        return None
    deck_dict = [asdict(o) for o in deck_pool]
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        json_file = d / "run.json"
        if not json_file.exists():
            continue
        data = json.loads(json_file.read_text())
        if 'deck_pool' not in data:
            continue
        if (data.get("completed_at") is None
                and data.get("abandoned_at") is None
                and data["deck_pool"] == deck_dict):
                # and data['version'] == version):
            return d
    return None

def move_winrates(run_dir: Path, gen: int) -> None:
    winrates_file = Path(MAGE_DIR) / f"WinRates.json"
    if winrates_file.exists():
        dest = run_dir / f"WinRates_gen{gen}.json"
        shutil.move(winrates_file, dest)
        print(f"Moved WinRates.json to {dest}")
    else:
        print("No WinRates.json found to move.")

# ─── main pipeline ───────────────────────────────────────────
def round_pipeline(run: RunRoundConfig, curriculum: CurriculumConfig,
                 base_game_yml: str = "configs/game.yml") -> None:
    # ── resume detection ──
    active = find_active_round_run(run.deck_pool, run.version)
    if active:
        ans = input(f"Active run found: {active.name}. Resume? [Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            run_dir = active
            start_gen = json.loads((active / "run.json").read_text())["current_gen"]
            print(f"Resuming from gen {start_gen}")
        else:
            update_run(active, abandoned_at=datetime.now().isoformat())
            # copy_starting_checkpoint(run)
            run_dir = create_round_dir(run)
            start_gen = 0
    else:
        # copy_starting_checkpoint(run)
        run_dir = create_round_dir(run)
        start_gen = 0

    # ── gen loop ──
    with open(base_game_yml) as f:
        base_game_cfg = yaml.safe_load(f)
    update_run(run_dir, threads=base_game_cfg['training'].get("threads", 1))
    timings = {}
    for gen in range(run.version+1, run.generations):
        print(f"\n========== GEN {gen} ==========")
        update_run(run_dir, current_gen=gen, stage="generate")
        settings = resolve_gen(curriculum, gen)
        curr_model_gen = gen - 1 if gen > 0 else run.start_from_version
        primary_sessions: dict[str, list[int]] = {}
        opponent_sessions: dict[str, list[int]] = {}
        matchups = combinations(run.deck_pool, 2)
        for ply, opp in matchups:
            print(f"\n[gen {gen}] {ply.deck} vs {opp.deck} ({opp.mode})")
            matchup_str = f"{ply.deck}_v{curr_model_gen}vs{opp.deck}_v{curr_model_gen}"
            # opp_ver = opp.version if opp.version is not None else latest_version(opp.deck)
            # if opp_ver is None:
            #     opp_ver = 1
            match_start = time.time()
            opp_offline = (opp.mode == "mcts") and not has_checkpoint(opp.deck, version=curr_model_gen)
            bootstrap = not has_checkpoint(ply.deck, version=curr_model_gen)
            primary_offline = bootstrap
            primary_sid = next_session_id(ply.deck)
            opponent_sid = next_session_id(opp.deck)
            primary_path = primary_file(ply.deck, gen, primary_sid, opp.deck)
            opponent_path = opponent_file(opp.deck, gen, opponent_sid, ply.deck)
            primary_path.parent.mkdir(parents=True, exist_ok=True)
            opponent_path.parent.mkdir(parents=True, exist_ok=True)
            game_yml = build_round_game_yml(
                base_game_yml, settings, run, ply, opp,
                primary_path, opponent_path, primary_offline, opp_offline,
            )
            if gen == 1 and (not (ply.deck =='DimirMidrange') and not (opp.deck == 'DimirMidrange')):
                print("Skipping matches for gen 1 except those involving DimirMidrange ")
                continue
            else:
                servers = []
                if not primary_offline:
                    servers.append(start_server(ply.deck, curr_model_gen, PRIMARY_PORT, run_dir))
                if not opp_offline:
                    servers.append(start_server(opp.deck, curr_model_gen, OPPONENT_PORT, run_dir))
                try:
                    launch_jvm(game_yml)
                finally:
                    for s in servers:
                        stop_server(s)
            match_length = time.time() - match_start
            print(f"Match completed in {match_length:.1f} seconds")
            timings[matchup_str] = match_length
            update_run(run_dir, stage="train", timing=timings)
            primary_sessions.setdefault(opp.deck, []).append(primary_sid)
            opponent_sessions.setdefault(opp.deck, []).append(opponent_sid)
        ## Move WinRates file from previous run:
        move_winrates(run_dir, gen)
        for ply in run.deck_pool:
            # ── analyze new data ──
            if run.training.analyze_dataset:
                update_run(run_dir, stage="analyze")
                run_dataset_stats(ply.deck, gen, "testing", run_dir, gen)

            # ── eval previous model on new data ──
            if run.training.eval_previous_model and gen > 1:
                update_run(run_dir, stage="eval")
                run_test(ply.deck, gen-1, run_dir, gen-1)
            
            # ── move testing → training ──
            update_run(run_dir, stage="move")
            # move_testing_to_training(ply.deck, gen)
            # ── archive out-of-window data ──
            update_run(run_dir, stage="train")
            data = json.loads((run_dir / "run.json").read_text())
            provisional = dict(data["gens"])
            provisional[str(gen)] = {"primary_sessions": primary_sessions}
            archived = archive_out_of_window(
                ply.deck, gen, provisional, gen, run.replay_buffer_gens,
            )

            # ── train ──
            try:
                print(f"Training model for deck {ply.deck}")
                epochs = EPOCHS_BOOTSTRAP if bootstrap else EPOCHS_ONLINE
                run_train(ply.deck, gen, epochs, use_checkpoint=not bootstrap,
                        run_dir=run_dir)
            finally:
                restore_from_archive(ply.deck, gen, archived)

        record_gen(run_dir, gen, settings, primary_sessions, opponent_sessions)

    update_run(run_dir, completed_at=datetime.now().isoformat(), stage="done")
    print(f"\n✓ Run complete: {run_dir.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="configs/run.yml")
    args = parser.parse_args()
    run_cfg, cur_cfg = load_all(args.run)
    round_pipeline(run_cfg, cur_cfg)