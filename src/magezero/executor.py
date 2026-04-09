import argparse
import logging
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

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    level_name = os.environ.get("MAGEZERO_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


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
        source_path = Path(source)
        try:
            file_and_timestamp = Path(source_path.parts[-2]) / source_path.name
            timestamp = source.parent.name
        except ValueError:
            file_and_timestamp = source.name

        timestamp_dir = destination_root / timestamp
        if not timestamp_dir.exists():
            timestamp_dir.mkdir(parents=True, exist_ok=True)
        target = destination_root / f"{file_and_timestamp}"
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
                if iteration == 1:
                    break
                else:
                    print('Running KrenkoMain with player deck {player_deck} and opponent deck {opp_deck}')
                wait_for_http(f"http://{args.server_host}:{args.server_port}/healthz", timeout_s=60)
                wait_for_http(f"http://{args.server_host}:{args.server_opponent_port}/healthz", timeout_s=60)

                ## This is an ugly way of handling this, should be cleaned up with configs
                update_model_weights(f"http://{args.server_host}:{args.server_port}/load", str(MAGEZERO_ROOT / "models" / Path(player_deck).stem / f"ver{iteration - 1}" / "model.pt"))
                update_model_weights(f"http://{args.server_host}:{args.server_opponent_port}/load", str(MAGEZERO_ROOT / "models" / Path(opp_deck).stem / f"ver{iteration - 1}" / "model.pt"))
                wait_for_http(f"http://{args.server_host}:{args.server_port}/healthz", timeout_s=60)
                wait_for_http(f"http://{args.server_host}:{args.server_opponent_port}/healthz", timeout_s=60)
                print("[server] ready", flush=True)
                print(f"[loop] iteration {iteration}/{args.iterations}: starting self-play", flush=True)
                args.deck_path = player_deck
                args.opp_path = opp_deck
                args.version = iteration
                run_command(build_krenko_command(args), cwd=MAGE_ROOT, env=os.environ.copy())
            server.terminate()
            server_opponent.terminate()
            iter_files = gather_hdf5_files(MAGE_ROOT / iter_output_dir)
            if not iter_files:
                raise RuntimeError("KrenkoMain completed without producing any new .hdf5 files.")
            training_dir = (MAGEZERO_ROOT / "data" / f"ver{iteration}" / "training").resolve()
            copied = copy_new_files(iter_files, training_dir, iteration)
            print(f"[loop] iteration {iteration}: copied {copied} training file(s) into {training_dir}", flush=True)
            
            for deck in DECKS:
                print(f"[loop] iteration {iteration}: training model", flush=True)
                if iteration == 1 and deck == "decks/JeskaiControl.dck":
                    print("One time skip")
                else:
                    train_env["MAGEZERO_DECK_NAME"] = Path(deck).stem
                    train_env["MAGEZERO_VER_NUMBER"] = str(iteration)
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

def generate_data() -> None:
    """Run a single batch of self-play data generation, optionally backed by a model server."""
    parser = argparse.ArgumentParser(
        description="Generate MTG self-play training data via XMage KrenkoMain."
    )
    parser.add_argument("--deck-path", required=True,
                        help="Player deck path relative to the mage repo (e.g. decks/IzzetElementals.dck).")
    parser.add_argument("--deck-name", default=None,
                        help="Deck name used for model lookup. Defaults to deck-path filename stem.")
    parser.add_argument("--opponent-deck", default=None,
                        help="Opponent deck path. Omit for a mirror match.")
    parser.add_argument("--opponent-name", default=None,
                        help="Opponent deck name. Defaults to opponent-deck filename stem.")
    parser.add_argument("--version", type=int, default=0, help="Model/data version number.")
    parser.add_argument("--games", type=int, default=18, help="Number of self-play games to generate.")
    parser.add_argument("--threads", type=int, default=8, help="XMage worker thread count.")
    parser.add_argument("--max-turns", type=int, default=40, help="Maximum turns per game.")
    parser.add_argument("--config-path", default="Mage.MageZero/config/krenko_config.yml",
                        help="KrenkoMain config path relative to the mage repo.")
    parser.add_argument("--mage-output-dir", default="data",
                        help="Folder within the mage repo where XMage writes HDF5 files.")
    parser.add_argument("--dest-dir", default=None,
                        help="Destination dir under MageZero. Defaults to data/ver{N}/training.")
    parser.add_argument("--offline", action="store_true",
                        help="Force offline mode (no inference server, even if a model exists).")
    parser.add_argument("--external-server", action="store_true",
                        help="Don't start the inference server; assume one is already running.")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=50052)
    parser.add_argument("--server-opponent-port", type=int, default=50053)
    parser.add_argument("--server-threads", type=int, default=6)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    _setup_logging()

    deck_name = args.deck_name or Path(args.deck_path).stem
    is_mirror = args.opponent_deck is None
    opp_deck_path = args.opponent_deck or args.deck_path
    opp_deck_name = args.opponent_name or Path(opp_deck_path).stem

    player_model = MAGEZERO_ROOT / "models" / deck_name / f"ver{args.version}" / "model.pt"
    opp_model = MAGEZERO_ROOT / "models" / opp_deck_name / f"ver{args.version}" / "model.pt"

    if args.offline:
        online = False
        logger.info("Mode: offline (forced via --offline)")
    elif not player_model.exists():
        online = False
        logger.info("Mode: offline (no model at %s)", player_model)
    elif not is_mirror and not opp_model.exists():
        online = False
        logger.info("Mode: offline (no opponent model at %s)", opp_model)
    else:
        online = True
        logger.info("Mode: online")
    logger.info("Player deck: %s, opponent deck: %s, mirror: %s", deck_name, opp_deck_name, is_mirror)

    krenko_args = argparse.Namespace(
        deck_path=args.deck_path,
        opp_path=opp_deck_path,
        config_path=args.config_path,
        games_per_test=args.games,
        number_of_tests=1,
        max_turns=args.max_turns,
        threads=args.threads,
        output_dir=args.mage_output_dir,
        version=args.version,
    )

    servers: list[subprocess.Popen] = []
    try:
        if online and not args.external_server:
            player_env = os.environ.copy()
            player_env["MAGEZERO_DECK_NAME"] = deck_name
            player_env["MAGEZERO_VER_NUMBER"] = str(args.version)
            player_env["MAGEZERO_SERVER_THREADS"] = str(args.server_threads)
            player_server_args = argparse.Namespace(
                python=args.python,
                server_host=args.server_host,
                server_port=args.server_port,
            )
            servers.append(start_server(player_server_args, player_env))
            wait_for_http(f"http://{args.server_host}:{args.server_port}/healthz", timeout_s=60)
            update_model_weights(
                f"http://{args.server_host}:{args.server_port}/load",
                str(player_model),
            )
            logger.info("Player inference server ready on :%d", args.server_port)

            if not is_mirror:
                opp_env = os.environ.copy()
                opp_env["MAGEZERO_DECK_NAME"] = opp_deck_name
                opp_env["MAGEZERO_VER_NUMBER"] = str(args.version)
                opp_env["MAGEZERO_SERVER_THREADS"] = str(args.server_threads)
                opp_server_args = argparse.Namespace(
                    python=args.python,
                    server_host=args.server_host,
                    server_port=args.server_opponent_port,
                )
                servers.append(start_server(opp_server_args, opp_env))
                wait_for_http(f"http://{args.server_host}:{args.server_opponent_port}/healthz", timeout_s=60)
                update_model_weights(
                    f"http://{args.server_host}:{args.server_opponent_port}/load",
                    str(opp_model),
                )
                logger.info("Opponent inference server ready on :%d", args.server_opponent_port)
        elif online and args.external_server:
            logger.info("Using external inference server (skipping startup)")

        logger.info("Running KrenkoMain with player deck %s and opponent deck %s",
                    args.deck_path, opp_deck_path)

        # Snapshot existing files so we copy only what's new this run.
        mage_iter_dir = MAGE_ROOT / args.mage_output_dir / f"ver{args.version}"
        existing = gather_hdf5_files(mage_iter_dir)

        run_command(build_krenko_command(krenko_args), cwd=MAGE_ROOT, env=os.environ.copy())

        new_files = sorted(gather_hdf5_files(mage_iter_dir) - existing)
        if not new_files:
            logger.warning("KrenkoMain produced no new .hdf5 files")
            return

        if args.dest_dir:
            dest = Path(args.dest_dir).resolve()
        else:
            dest = (MAGEZERO_ROOT / "data" / f"ver{args.version}" / "training").resolve()
        copied = copy_new_files(new_files, dest, args.version)
        logger.info("Copied %d HDF5 file(s) into %s", copied, dest)

    finally:
        for srv in servers:
            srv.terminate()
        for srv in servers:
            try:
                srv.wait(timeout=15)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait(timeout=15)


def train_cli() -> None:
    """CLI wrapper around magezero.train.train that exposes common params as flags.

    Flags translate to MAGEZERO_* env vars before train.py is imported, so the
    module-level constants in train.py pick them up. Omitted flags fall through
    to the existing env-var defaults.
    """
    parser = argparse.ArgumentParser(
        description="Train a MageZero model on collected HDF5 data."
    )
    parser.add_argument("--deck-name", default=None, help="Override MAGEZERO_DECK_NAME.")
    parser.add_argument("--version", type=int, default=None, help="Override MAGEZERO_VER_NUMBER.")
    parser.add_argument("--epochs", type=int, default=None, help="Override MAGEZERO_EPOCH_COUNT.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override MAGEZERO_BATCH_SIZE.")
    parser.add_argument("--use-previous-model", action="store_true",
                        help="Set MAGEZERO_USE_PREVIOUS_MODEL=1.")
    parser.add_argument("--train-opponent-head", action="store_true",
                        help="Set MAGEZERO_TRAIN_OPPONENT_HEAD=1.")
    args = parser.parse_args()
    _setup_logging()

    if args.deck_name is not None:
        os.environ["MAGEZERO_DECK_NAME"] = args.deck_name
    if args.version is not None:
        os.environ["MAGEZERO_VER_NUMBER"] = str(args.version)
    if args.epochs is not None:
        os.environ["MAGEZERO_EPOCH_COUNT"] = str(args.epochs)
    if args.batch_size is not None:
        os.environ["MAGEZERO_BATCH_SIZE"] = str(args.batch_size)
    if args.use_previous_model:
        os.environ["MAGEZERO_USE_PREVIOUS_MODEL"] = "1"
    if args.train_opponent_head:
        os.environ["MAGEZERO_TRAIN_OPPONENT_HEAD"] = "1"

    # Lazy import: train.py reads MAGEZERO_* env vars at module load, so the
    # import has to happen *after* we've set them above.
    from magezero.train import train as run_train
    run_train()


if __name__ == "__main__":
    main()
