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

# Test locally with S3 upload
docker run --rm --platform linux/amd64 -v ~/.aws:/root/.aws:ro -e S3_BUCKET=batch-s3-writer-083665380715 -e S3_PREFIX=batch-output -e DECK_PATH=decks/MonoRAggro.dck -e VERSION=99 -e GAMES=1 -e THREADS=1 -e MAX_TURNS=5 -e OFFLINE=1 magezero-batch

# Tag and push to ECR (replace TAG)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 083665380715.dkr.ecr.us-east-1.amazonaws.com
docker tag magezero-batch 083665380715.dkr.ecr.us-east-1.amazonaws.com/magezero-batch:TAG
docker push 083665380715.dkr.ecr.us-east-1.amazonaws.com/magezero-batch:TAG

# Submit 1 job to Batch
aws batch submit-job --job-name magezero-test --job-queue batch-s3-writer-queue --job-definition magezero-batch-job --container-overrides '{"environment": [{"name": "DECK_PATH", "value": "decks/MonoRAggro.dck"}, {"name": "VERSION", "value": "0"}]}'

# Submit N jobs to Batch
for i in $(seq 1 100); do aws batch submit-job --job-name "magezero-game-${i}" --job-queue batch-s3-writer-queue --job-definition magezero-batch-job --container-overrides '{"environment": [{"name": "DECK_PATH", "value": "decks/MonoRAggro.dck"}, {"name": "VERSION", "value": "0"}]}' --query 'jobId' --output text; done

# Monitor jobs
aws batch list-jobs --job-queue batch-s3-writer-queue --job-status RUNNING --query 'length(jobSummaryList)'
aws batch list-jobs --job-queue batch-s3-writer-queue --job-status SUCCEEDED --query 'length(jobSummaryList)'

# Check S3 output
aws s3 ls s3://batch-s3-writer-083665380715/batch-output/ver0/training/ --recursive | wc -l

# Download and inspect data
aws s3 cp s3://batch-s3-writer-083665380715/batch-output/ver0/training/ /tmp/batch-data/ --recursive
uv run inspect-data /tmp/batch-data --samples 3
