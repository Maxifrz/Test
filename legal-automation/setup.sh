#!/usr/bin/env bash
# First-time setup script for Legal Automation Platform
# Run once on the host machine before docker compose up

set -euo pipefail

echo "=== Legal Automation Platform — Setup ==="

# Secrets liegen bewusst AUSSERHALB der Docker-Build-Kontexte (backend/, frontend/),
# damit `COPY . .` den privaten JWT-Schlüssel niemals in ein Image backen kann.
SECRETS_DIR="secrets"
mkdir -p "${SECRETS_DIR}"
chmod 700 "${SECRETS_DIR}"

# 1. Generate JWT key pair
if [ ! -f "${SECRETS_DIR}/jwt_private.pem" ]; then
    echo "Generating JWT RSA-4096 key pair..."
    openssl genrsa -out "${SECRETS_DIR}/jwt_private.pem" 4096
    openssl rsa -in "${SECRETS_DIR}/jwt_private.pem" -pubout -out "${SECRETS_DIR}/jwt_public.pem"
    # 0644: der Container läuft als non-root 'appuser' und muss den Key lesen
    # können; der Schutz kommt aus dem 0700-Verzeichnis, nicht aus der Datei.
    chmod 644 "${SECRETS_DIR}/jwt_private.pem" "${SECRETS_DIR}/jwt_public.pem"
    echo "  -> ${SECRETS_DIR}/jwt_private.pem + jwt_public.pem created"
else
    echo "  -> JWT keys already exist, skipping"
fi

# Migration älterer Installationen: Keys lagen früher in backend/ (Build-Kontext!)
if [ -f backend/jwt_private.pem ]; then
    echo "  !! WARNUNG: backend/jwt_private.pem gefunden (alter Pfad, lag im Docker-Build-Kontext)."
    echo "     Verschiebe nach ${SECRETS_DIR}/ und rotiere den Schlüssel, falls je ein Image gebaut wurde."
    mv backend/jwt_private.pem backend/jwt_public.pem "${SECRETS_DIR}/" 2>/dev/null || true
fi

# 2. Generate secrets for .env
if [ ! -f .env ]; then
    cp .env.example .env
    SECRET_KEY=$(openssl rand -hex 32)
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || echo "GENERATE_MANUALLY")
    DB_PASSWORD=$(openssl rand -hex 16)
    REDIS_PASSWORD=$(openssl rand -hex 16)
    PII_INDEX_KEY=$(openssl rand -hex 32)

    sed -i "s/^SECRET_KEY=$/SECRET_KEY=${SECRET_KEY}/" .env
    sed -i "s/^ENCRYPTION_KEYS=$/ENCRYPTION_KEYS=${FERNET_KEY}/" .env
    sed -i "s/^PII_INDEX_KEY=$/PII_INDEX_KEY=${PII_INDEX_KEY}/" .env
    sed -i "s/CHANGE_ME_DB_PASSWORD/${DB_PASSWORD}/g" .env
    sed -i "s/CHANGE_ME_REDIS_PASSWORD/${REDIS_PASSWORD}/g" .env

    if [ "${FERNET_KEY}" = "GENERATE_MANUALLY" ]; then
        echo "  !! python3 + cryptography nicht verfügbar — ENCRYPTION_KEYS manuell setzen!"
    fi

    echo "  -> .env created with generated secrets"
    echo ""
    echo "  IMPORTANT: Review .env and fill in:"
    echo "    - ALLOWED_HOSTS (your server's hostname)"
    echo "    - IMAP_*/SMTP_* (email account settings)"
    echo "    - BUNDESLAND (your state for holiday calculation)"
    echo "    - DSGVO_CONTACT_EMAIL"
else
    echo "  -> .env already exists, skipping"
    # Bestehende Installationen ohne Redis-Passwort / PII-Index nachziehen
    if ! grep -q "^REDIS_PASSWORD=" .env; then
        echo "  !! .env hat kein REDIS_PASSWORD. Redis startet mit --requirepass —"
        echo "     ohne Passwort in REDIS_URL/CELERY_* schlägt jede Verbindung mit NOAUTH fehl."
    fi
    if ! grep -q "^PII_INDEX_KEY=" .env; then
        echo "  !! .env hat keinen PII_INDEX_KEY (Blind-Index für verschlüsselte Suche)."
        echo "     Ergänzen mit: echo \"PII_INDEX_KEY=\$(openssl rand -hex 32)\" >> .env"
    fi
fi

# 3. Generate self-signed TLS cert for development
if [ ! -f nginx/ssl/server.crt ]; then
    echo "Generating self-signed TLS certificate (replace with real cert in production)..."
    mkdir -p nginx/ssl
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/server.key \
        -out nginx/ssl/server.crt \
        -subj "/C=DE/ST=Bayern/L=Muenchen/O=Kanzlei/CN=localhost"
    echo "  -> nginx/ssl/server.crt + server.key created"
    echo "  -> IMPORTANT: Replace with a proper certificate before production use!"
else
    echo "  -> TLS certificates already exist, skipping"
fi

# 4. Create storage directories
mkdir -p storage/{documents,transcriptions,exports,emails,dsgvo}
echo "  -> storage/ directories created"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Review and complete .env"
echo "  2. docker compose up -d"
echo "  3. Login at https://localhost with:"
echo "     Email:    admin@kanzlei.local"
echo "     Password: Admin1234!  (Wechsel wird beim ersten Login erzwungen)"
