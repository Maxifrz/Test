#!/usr/bin/env bash
#
# Restore-Probe: spielt das jüngste Backup in eine WEGWERF-Datenbank ein und
# prüft, ob daraus ein arbeitsfähiger Stand entsteht.
#
# Warum automatisiert: das Betriebshandbuch verlangt eine Restore-Probe zu
# Recht — manuell wird sie erfahrungsgemäß nach dem Go-Live nie wieder
# durchgeführt. Ein Backup, das nie zurückgespielt wurde, ist eine Annahme,
# kein Backup.
#
# Aufruf (z. B. wöchentlich per systemd-timer):
#   /usr/local/bin/restore-probe.sh /var/backups/legal
#
# Rückgabe: 0 = Probe bestanden, sonst Fehler (für Monitoring auswertbar).

set -euo pipefail

BACKUP_DIR="${1:-/var/backups/legal}"
PROBE_DB="legal_restore_probe_$(date +%s)"
COMPOSE_DIR="${COMPOSE_DIR:-/opt/legal-automation}"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { log "FEHLGESCHLAGEN: $*"; exit 1; }

cleanup() {
    log "Räume Probe-Datenbank ${PROBE_DB} ab"
    docker compose -f "${COMPOSE_DIR}/docker-compose.yml" exec -T postgres \
        psql -U "${POSTGRES_USER:-legalapp}" -d postgres \
        -c "DROP DATABASE IF EXISTS ${PROBE_DB}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --- 1) Jüngsten Dump finden ---
DUMP=$(find "${BACKUP_DIR}" -name 'db-*.dump' -type f -printf '%T@ %p\n' 2>/dev/null \
       | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "${DUMP}" ] || fail "Kein Dump unter ${BACKUP_DIR} gefunden"

DUMP_AGE_H=$(( ( $(date +%s) - $(stat -c %Y "${DUMP}") ) / 3600 ))
log "Jüngster Dump: ${DUMP} (${DUMP_AGE_H} h alt)"
# Ein Backup, das älter als 48 h ist, hat entweder nicht gelaufen oder ist
# stillschweigend gescheitert — beides ist ein Befund.
[ "${DUMP_AGE_H}" -le 48 ] || fail "Dump ist ${DUMP_AGE_H} h alt (erwartet: <= 48 h)"

# --- 2) In eine Wegwerf-Datenbank einspielen ---
log "Lege Probe-Datenbank ${PROBE_DB} an"
docker compose -f "${COMPOSE_DIR}/docker-compose.yml" exec -T postgres \
    psql -U "${POSTGRES_USER:-legalapp}" -d postgres \
    -c "CREATE DATABASE ${PROBE_DB}" >/dev/null

log "Spiele Dump ein"
docker compose -f "${COMPOSE_DIR}/docker-compose.yml" exec -T postgres \
    pg_restore -U "${POSTGRES_USER:-legalapp}" -d "${PROBE_DB}" --no-owner --exit-on-error \
    < "${DUMP}" >/dev/null || fail "pg_restore ist gescheitert"

# --- 3) Inhaltliche Prüfungen ---
# Ein technisch erfolgreicher Restore sagt noch nichts darüber, ob die Daten
# brauchbar sind. Deshalb wird geprüft, was im Ernstfall zählt.
query() {
    docker compose -f "${COMPOSE_DIR}/docker-compose.yml" exec -T postgres \
        psql -U "${POSTGRES_USER:-legalapp}" -d "${PROBE_DB}" -tAc "$1"
}

TABLES=$(query "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")
log "Tabellen: ${TABLES}"
[ "${TABLES}" -ge 30 ] || fail "Nur ${TABLES} Tabellen (erwartet: >= 30)"

USERS=$(query "SELECT count(*) FROM users WHERE deleted_at IS NULL")
log "Aktive Benutzer: ${USERS}"
[ "${USERS}" -ge 1 ] || fail "Keine Benutzer im Restore — Anmeldung wäre unmöglich"

MATTERS=$(query "SELECT count(*) FROM matters WHERE deleted_at IS NULL")
log "Akten: ${MATTERS}"

# Migrationsstand: ein Restore auf altem Schema würde beim Start scheitern
REVISION=$(query "SELECT version_num FROM alembic_version")
log "Alembic-Revision: ${REVISION}"
[ -n "${REVISION}" ] || fail "Keine alembic_version — Schema unvollständig"

# Unveränderlichkeit des Audit-Logs muss den Restore überleben
TRIGGER=$(query "SELECT count(*) FROM pg_trigger WHERE tgname='audit_log_immutable'")
[ "${TRIGGER}" -ge 1 ] || fail "Audit-Log-Trigger fehlt nach dem Restore"
log "Audit-Log-Trigger vorhanden"

# Verschlüsselte Felder müssen mit dem AKTUELLEN Schlüssel lesbar sein.
# Ein Backup ohne passenden ENCRYPTION_KEYS ist wertlos — genau davor warnt
# das Betriebshandbuch, und genau das prüft dieser Schritt.
ENCRYPTED=$(query "SELECT count(*) FROM clients WHERE email IS NOT NULL AND deleted_at IS NULL")
if [ "${ENCRYPTED}" -gt 0 ]; then
    log "Prüfe Entschlüsselung von ${ENCRYPTED} Mandanten-Adressen"
    docker compose -f "${COMPOSE_DIR}/docker-compose.yml" exec -T backend \
        python -c "
import asyncio, os, sys
os.environ['DATABASE_URL'] = os.environ['DATABASE_URL'].rsplit('/', 1)[0] + '/${PROBE_DB}'
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from app.models.client import Client

async def main():
    engine = create_async_engine(os.environ['DATABASE_URL'])
    async with async_sessionmaker(engine)() as db:
        rows = (await db.execute(select(Client).limit(5))).scalars().all()
        for r in rows:
            _ = r.email, r.phone, r.notes   # loest die Entschluesselung aus
    await engine.dispose()
    print(f'{len(rows)} Datensaetze entschluesselt')

asyncio.run(main())
" || fail "Verschlüsselte Felder sind mit dem aktuellen ENCRYPTION_KEYS nicht lesbar"
fi

log "Restore-Probe BESTANDEN (Dump ${DUMP})"
