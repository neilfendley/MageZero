import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from tqdm import tqdm
from itertools import combinations


MODULE_DIR = Path(__file__).resolve().parent
MAGEZERO_ROOT = MODULE_DIR.parent.parent
WORKSPACE_ROOT = MAGEZERO_ROOT.parent
MAGE_ROOT = WORKSPACE_ROOT / "mage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MageZero self-play/training/reload loops with KrenkoMain."
    )
    parser.add_argument("--deck-path", default='decks/IzzetElementals.dck', help="Deck path relative to the mage repo, e.g. decks/IzzetElementals.dck")
    parser.add_argument("--deck-name", help="Deck name used under MageZero/data and MageZero/models. Defaults to the deck filename stem.")
    parser.add_argument("--version", type=int, default=0, help="MageZero model/data version number.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of self-play/train cycles to run.")
    parser.add_argument("--games-per-test", type=int, default=32, help="Games per KrenkoMain batch.")
    parser.add_argument("--number-of-tests", type=int, default=1, help="How many KrenkoMain batches to run per cycle.")
    parser.add_argument("--max-turns", type=int, default=40, help="Maximum turns per game.")
    parser.add_argument("--threads", type=int, default=20, help="XMage self-play worker thread count.")
    parser.add_argument("--server-host", default="127.0.0.1", help="Inference server host.")
    parser.add_argument("--server-port", type=int, default=50052, help="Inference server port.")
    parser.add_argument("--server-threads", type=int, default=6, help="Waitress thread count for the inference server.")
    parser.add_argument("--config-path", default="Mage.MageZero/config/krenko_config.yml", help="KrenkoMain config path relative to the mage repo.")
    parser.add_argument("--player-a-output-dir", default="data/selfplay/playerA", help="Krenko output dir for player A, relative to the mage repo.")
    parser.add_argument("--player-b-output-dir", default="data/selfplay/playerB", help="Krenko output dir for player B, relative to the mage repo.")
    parser.add_argument("--train-epochs", type=int, help="Override MAGEZERO_EPOCH_COUNT for each training run.")
    parser.add_argument("--train-opponent-head", action="store_true", help="Set MAGEZERO_TRAIN_OPPONENT_HEAD=1 during training.")
    parser.add_argument("--output-dir", default="data", help="Folder within mage to save games to")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for server and training.")
    return parser.parse_args()


def gather_hdf5_files(*roots: Path) -> set[Path]:
    files = set()
    for root in roots:
        if root.exists():
            files.update(path.resolve() for path in root.rglob("*.hdf5"))
    return files


def wait_for_http(url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {url}")


def post(url: str) -> None:
    request = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if not (200 <= response.status < 300):
                raise RuntimeError(f"POST {url} failed with HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}") from exc


def run_command(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    print(f"[cmd] cwd={cwd}", flush=True)
    # subprocess.run(cmd, cwd=cwd, check=True, env=env,stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    # subprocess.run(cmd, cwd=cwd, check=True, env=env)
    subprocess.run(cmd, cwd=cwd, check=True, shell=True, env=env)



def start_server(args: argparse.Namespace, env: dict[str, str]) -> subprocess.Popen:
    cmd = [
        args.python,
        "-m",
        "waitress",
        f"--host={args.server_host}",
        f"--port={args.server_port}",
        f"--threads={args.server_threads}",
        "magezero.server:app",
    ]
    print(f"[server] starting on http://{args.server_host}:{args.server_port}", flush=True)
    return subprocess.Popen(cmd, cwd=MAGEZERO_ROOT, env=env)


def build_krenko_command(args: argparse.Namespace) -> list[str]:
    exec_args = [
        "--config", args.config_path,
        "--player-deck", args.deck_path,
        "--opponent-deck", args.opp_path,
        "--games-per-test", str(args.games_per_test),
        "--number-of-tests", str(args.number_of_tests),
        "--max-turns", str(args.max_turns),
        "--threads", str(args.threads),
        "--player-a-type", "mcts",
        "--player-b-type", "mcts",
        "--output-dir", args.output_dir,
    ]
    exec_args_str = " ".join(exec_args)
    return [
        "mvn",
        "-pl",
        "Mage.Tests",
        # "-am",
        # "test-compile",
        "exec:java",
        "-Dexec.classpathScope=test",
        "-Dexec.mainClass=org.mage.test.AI.KrenkoMain",
        f"-Dexec.args={exec_args_str}",
    ]


def copy_new_files(new_files: list[Path], destination_root: Path, iteration: int) -> int:
    destination_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in sorted(new_files):
        try:
            relative = source.relative_to(MAGE_ROOT)
            relative_key = "__".join(relative.parts)
        except ValueError:
            relative_key = source.name
        target = destination_root / f"iter{iteration:04d}__{relative_key}"
        shutil.copy2(source, target)
        copied += 1
    return copied


def round_robin() -> None:
    args = parse_args()
    deck_name = args.deck_name or Path(args.deck_path).stem

    
    decks = ["decks/DimirMidrange.dck",
            "decks/JeskaiControl.dck",
            "decks/MonoGLandfall.dck",
            "decks/MonoRAggro.dck",
            "decks/BWBats.dck",
            ]
    comb = combinations(decks, 2)
    for player_deck, opp_deck in comb:
        print(f'Running KrenkoMain with player deck {player_deck} and opponent deck {opp_deck}')
        for num_threads in [2,4,8,10]:
            args.threads = num_threads
            try:
                args.deck_path = player_deck
                args.opp_path = opp_deck
                run_command(build_krenko_command(args), cwd=MAGE_ROOT,env=os.environ.copy())
            except Exception as e:
                print(f'something failed with decks {player_deck} {opp_deck}')

def one_deck_per_model() -> None:
    args = parse_args()
    deck_name = args.deck_name or Path(args.deck_path).stem

    player_a_output_dir = (MAGE_ROOT / args.player_a_output_dir).resolve()
    player_b_output_dir = (MAGE_ROOT / args.player_b_output_dir).resolve()
    training_dir = (MAGEZERO_ROOT / "data" / deck_name / f"ver{args.version}" / "training").resolve()


    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(MAGEZERO_ROOT / "src") + os.pathsep + base_env.get("PYTHONPATH", "")
    base_env["MAGEZERO_DECK_NAME"] = deck_name
    base_env["MAGEZERO_VER_NUMBER"] = str(args.version)
    base_env["MAGEZERO_SERVER_HOST"] = args.server_host
    base_env["MAGEZERO_SERVER_PORT"] = str(args.server_port) 

    train_env = base_env.copy()
    train_env["MAGEZERO_USE_PREVIOUS_MODEL"] = "1"
    if args.train_epochs is not None:
        train_env["MAGEZERO_EPOCH_COUNT"] = str(args.train_epochs)
    if args.train_opponent_head:
        train_env["MAGEZERO_TRAIN_OPPONENT_HEAD"] = "1"

    args = parse_args()
    server = start_server(args, base_env)
    try:
        wait_for_http(f"http://{args.server_host}:{args.server_port}/healthz", timeout_s=60)
        print("[server] ready", flush=True)

        for iteration in range(1, args.iterations + 1):
            print(f"[loop] iteration {iteration}/{args.iterations}: starting self-play", flush=True)
            before = gather_hdf5_files(player_a_output_dir, player_b_output_dir)
            run_command(build_krenko_command(args), cwd=MAGE_ROOT, env=os.environ.copy())

            after = gather_hdf5_files(player_a_output_dir, player_b_output_dir)
            new_files = sorted(after - before)
            if not new_files:
                raise RuntimeError("KrenkoMain completed without producing any new .hdf5 files.")

            copied = copy_new_files(new_files, training_dir, iteration)
            print(f"[loop] iteration {iteration}: copied {copied} training file(s) into {training_dir}", flush=True)

            print(f"[loop] iteration {iteration}: training model", flush=True)
            run_command([args.python, "-m", "magezero.train"], cwd=MAGEZERO_ROOT, env=train_env)

            print(f"[loop] iteration {iteration}: reloading inference server", flush=True)
            post(f"http://{args.server_host}:{args.server_port}/reload")

        print("[loop] complete", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=15)


if __name__ == "__main__":
    main()
