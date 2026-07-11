#!/usr/bin/env bash
# First-time setup script for Legal Automation Platform
# Run once on the host machine before docker compose up

set -euo pipefail

echo "=== Legal Automation Platform — Setup ==="

# 1. Generate JWT key pair
if [ ! -f backend/jwt_private.pem ]; then
    echo "Generating JWT RSA-4096 key pair..."
    openssl genrsa -out backend/jwt_private.pem 4096
    openssl rsa -in backend/jwt_private.pem -pubout -out backend/jwt_public.pem
    chmod 600 backend/jwt_private.pem
    echo "  -> backend/jwt_private.pem + backend/jwt_public.pem created"
else
    echo "  -> JWT keys already exist, skipping"
fi

# 2. Generate secrets for .env
if [ ! -f .env ]; then
    cp .env.example .env
    SECRET_KEY=$(openssl rand -hex 32)
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || echo "GENERATE_MANUALLY")
    DB_PASSWORD=$(openssl rand -hex 16)
    REDIS_PASSWORD=$(openssl rand -hex 16)

    sed -i "s/^SECRET_KEY=$/SECRET_KEY=${SECRET_KEY}/" .env
    sed -i "s/^ENCRYPTION_KEYS=$/ENCRYPTION_KEYS=${FERNET_KEY}/" .env
    sed -i "s/CHANGE_ME_DB_PASSWORD/${DB_PASSWORD}/g" .env
    # Redis-Passwort überall ersetzen (REDIS_PASSWORD + die drei redis://-URLs)
    sed -i "s/changeme_redis/${REDIS_PASSWORD}/g" .env

    echo "  -> .env created with generated secrets"
    echo ""
    echo "  IMPORTANT: Review .env and fill in:"
    echo "    - ALLOWED_HOSTS (your server's hostname)"
    echo "    - IMAP_*/SMTP_* (email account settings)"
    echo "    - BUNDESLAND (your state for holiday calculation)"
    echo "    - DSGVO_CONTACT_EMAIL"
else
    echo "  -> .env already exists, skipping"
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
mkdir -p storage/{documents,transcriptions,exports}
echo "  -> storage/ directories created"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Review and complete .env"
echo "  2. docker compose up -d"
echo "  3. Login at https://localhost with:"
echo "     Email:    admin@kanzlei.local"
echo "     Password: Admin1234!  (CHANGE THIS IMMEDIATELY)"
