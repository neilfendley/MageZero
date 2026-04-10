FROM magezero-base:latest

WORKDIR /app

# Shaded jar — built locally via: mvn package -T 1C -pl Mage.Tests -am -DskipTests
# Preserve mage/ directory structure so relative paths (decks/, config, output) work
COPY mage/Mage.Tests/target/mage-tests.jar /app/krenko.jar
COPY mage/decks/ /app/mage/decks/
COPY mage/Mage.MageZero/config/ /app/mage/Mage.MageZero/config/
COPY MageZero/ /app/MageZero/

WORKDIR /app/MageZero
RUN pip install --no-cache-dir --no-deps -e .

ENV KRENKO_JAR=/app/krenko.jar

ENTRYPOINT ["python", "batch/generate_and_upload.py"]
