import argparse
from math import comb
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

DECKS = [
    "decks/MonoRAggro.dck",
    "decks/IzzetElementals.dck",
    "decks/JeskaiControl.dck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MageZero self-play/training/reload loops with KrenkoMain."
    )
    # parser.add_argument("--deck-path", default='decks/IzzetElementals.dck', help="Deck path relative to the mage repo, e.g. decks/IzzetElementals.dck")
    # parser.add_argument("--deck-name", help="Deck name used under MageZero/data and MageZero/models. Defaults to the deck filename stem.")
    parser.add_argument("--version", type=int, default=0, help="MageZero model/data version number.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of self-play/train cycles to run.")
    parser.add_argument("--games-per-test", type=int, default=18, help="Games per KrenkoMain batch.")
    parser.add_argument("--number-of-tests", type=int, default=1, help="How many KrenkoMain batches to run per cycle.")
    parser.add_argument("--max-turns", type=int, default=40, help="Maximum turns per game.")
    parser.add_argument("--threads", type=int, default=18, help="XMage self-play worker thread count.")
    parser.add_argument("--server-host", default="127.0.0.1", help="Inference server host.")
    parser.add_argument("--server-port", type=int, default=50052, help="Inference server port.")
    parser.add_argument("--server-opponent-port", type=int, default=50053, help="Inference server port for opponent head.")
    parser.add_argument("--server-threads", type=int, default=6, help="Waitress thread count for the inference server.")
    parser.add_argument("--config-path", default="Mage.MageZero/config/krenko_config.yml", help="KrenkoMain config path relative to the mage repo.")
    parser.add_argument("--train-epochs", default = 10, type=int, help="Override MAGEZERO_EPOCH_COUNT for each training run.")
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


def update_model_weights(url: str, model_path: str) -> None:
    import json
    data = json.dumps({"path": model_path}).encode('utf-8')
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header('Content-Type', 'application/json')
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
    import platform

    current_os = platform.system()

    if current_os == "Windows":
        print("Running on Windows")
        shell = True
    elif current_os == "Darwin":
        print("Running on macOS")
        shell = False
    else:
        shell = False
    subprocess.run(cmd, cwd=cwd, check=True, shell=shell, env=env)



def start_server(arg_server: argparse.Namespace, env: dict[str, str]) -> subprocess.Popen:
    cmd = [
        arg_server.python,
        "-m",
        "waitress",
        f"--host={arg_server.server_host}",
        f"--port={arg_server.server_port}",
        "magezero.server:app",
    ]
    print(f"[server] starting on http://{arg_server.server_host}:{arg_server.server_port}", flush=True)
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
        "--version", str(args.version),
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
    """ Run one batch of self play for generating games without a server running. Useful for generating initial training data. """
    args = parse_args()
    comb = combinations(DECKS, 2)
    for player_deck, opp_deck in comb:
        print(f'Running KrenkoMain with player deck {player_deck} and opponent deck {opp_deck}')
        try:
            args.deck_path = player_deck
            args.opp_path = opp_deck
            run_command(build_krenko_command(args), cwd=MAGE_ROOT,env=os.environ.copy())
        except Exception as e:
            print(f'something failed with decks {player_deck} {opp_deck}')
        break
        
def one_deck_per_model() -> None:
    args = parse_args()

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(MAGEZERO_ROOT / "src") + os.pathsep + base_env.get("PYTHONPATH", "")
    base_env["MAGEZERO_DECK_NAME"] = Path(DECKS[0]).stem
    base_env["MAGEZERO_VER_NUMBER"] = str(args.version)
    base_env["MAGEZERO_SERVER_HOST"] = args.server_host
    base_env["MAGEZERO_SERVER_PORT"] = str(args.server_port) 
    base_env["MAGEZERO_SERVER_THREADS"] = str(args.server_threads)

    train_env = base_env.copy()
    train_env["MAGEZERO_USE_PREVIOUS_MODEL"] = "1"
    train_env["MAGEZERO_EPOCH_COUNT"] = str(args.train_epochs)
    if args.train_opponent_head:
        train_env["MAGEZERO_TRAIN_OPPONENT_HEAD"] = "1"
    
    for deck in DECKS:
        deck_name = Path(deck).stem
        train_env["MAGEZERO_DECK_NAME"] = deck_name
        train_env["MAGEZERO_VER_NUMBER"] = str(args.version)
        if not Path(f"models/{deck_name}/ver{args.version}/model.pt").exists():
            run_command([args.python, "-m", "magezero.train"], cwd=MAGEZERO_ROOT, env=train_env)

    try:
        for iteration in range(args.version + 1, args.version + args.iterations + 1):
            server_args = args_copy = argparse.Namespace(**vars(args))
            server_args.server_port = args.server_port
            base_env["MAGEZERO_DECK_NAME"] = Path(DECKS[0]).stem
            base_env["MAGEZERO_VER_NUMBER"] = str(iteration - 1)
            server = start_server(server_args, base_env)
            server_args_opponent = argparse.Namespace(**vars(args))
            server_args_opponent.server_port = args.server_opponent_port
            server_opp_env = base_env.copy()
            server_opp_env["MAGEZERO_DECK_NAME"] = Path(DECKS[1]).stem
            # breakpoint()
            server_opponent = start_server(server_args_opponent, server_opp_env)
            iter_output_dir = Path(args.output_dir) / f"ver{iteration}"
            comb = combinations(DECKS, 2)

            for player_deck, opp_deck in comb:
                wait_for_http(f"http://{args.server_host}:{args.server_port}/healthz", timeout_s=60)
                wait_for_http(f"http://{args.server_host}:{args.server_opponent_port}/healthz", timeout_s=60)

                ## This is an ugly way of handling this, should be cleaned up with configs
                update_model_weights(f"http://{args.server_host}:{args.server_port}/load", str(MAGEZERO_ROOT / "models" / Path(player_deck).stem / f"ver{iteration - 1}" / "model.pt"))
                update_model_weights(f"http://{args.server_host}:{args.server_opponent_port}/load", str(MAGEZERO_ROOT / "models" / Path(opp_deck).stem / f"ver{iteration - 1}" / "model.pt"))
                wait_for_http(f"http://{args.server_host}:{args.server_port}/healthz", timeout_s=60)
                wait_for_http(f"http://{args.server_host}:{args.server_opponent_port}/healthz", timeout_s=60)
                print("[server] ready", flush=True)
                print(f"[loop] iteration {iteration}/{args.iterations}: starting self-play", flush=True)
                before = gather_hdf5_files(iter_output_dir)
                args.deck_path = player_deck
                args.opp_path = opp_deck
                args.version = iteration
                run_command(build_krenko_command(args), cwd=MAGE_ROOT, env=os.environ.copy())
                after = gather_hdf5_files(iter_output_dir)
                new_files = sorted(after - before)
                if not new_files:
                    raise RuntimeError("KrenkoMain completed without producing any new .hdf5 files.")
                training_dir = (MAGEZERO_ROOT / "data" / f"ver{iteration}" / "training").resolve()
                copied = copy_new_files(new_files, training_dir, iteration)
                print(f"[loop] iteration {iteration}: copied {copied} training file(s) into {training_dir}", flush=True)
            for deck in DECKS:
                print(f"[loop] iteration {iteration}: training model", flush=True)
                train_env["MAGEZERO_DECK_NAME"] = deck_name
                train_env["MAGEZERO_VER_NUMBER"] = str(args.iteration)
                run_command([args.python, "-m", "magezero.train"], cwd=MAGEZERO_ROOT, env=train_env)
                print(f"[loop] iteration {iteration}: updating model weights", flush=True)
    
    finally:
        print("[loop] complete", flush=True)
        server.terminate()
        server_opponent.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=15)

if __name__ == "__main__":
    main()
