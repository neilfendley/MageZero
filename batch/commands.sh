# Build shaded KrenkoMain jar (run from mtg/mage/)
mvn package -T 1C -pl Mage.Tests -am -DskipTests

# Update pinned CPU-only Python deps (run from mtg/MageZero/)
uv export --no-dev --no-hashes --no-emit-project 2>/dev/null | grep -v 'nvidia\|cuda-\|triton\|^#\|^$' | sed '/^    #/d' > batch/requirements-cpu.txt

# Build base image (run from mtg/)
docker build --platform linux/amd64 -f MageZero/batch/Dockerfile.base -t magezero-base .

# Build batch image (run from mtg/)
docker build --platform linux/amd64 -f MageZero/batch/Dockerfile -t magezero-batch .

# Test locally — offline, 1 game
docker run --rm --platform linux/amd64 --entrypoint sh magezero-batch -c "generate-data --deck-path decks/MonoRAggro.dck --version 99 --games 1 --threads 1 --max-turns 1 --offline"
