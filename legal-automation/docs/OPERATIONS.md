# Betriebshandbuch (Operations)

Zielgruppe: IT-Verantwortliche der Kanzlei bzw. beauftragter Dienstleister.
Ergänzt `README.md` (Setup) und `docs/dsgvo/TOM.md` (technisch-organisatorische
Maßnahmen). **Backups enthalten personenbezogene Daten — sie unterliegen
denselben Schutzmaßnahmen wie das Live-System (Verschlüsselung, Zugriff,
Löschkonzept).**

---

## 1. Was gesichert werden muss

| Bestandteil | Ort | Inhalt |
|---|---|---|
| PostgreSQL | Docker-Volume `pgdata` | Alle Stammdaten, Akten, E-Mails, Audit-Log, KI-Wissensgraph |
| Dateispeicher | `./storage` (Bind-Mount) | Verschlüsselte Originale: Transkriptions-Audio, E-Mail-Anhänge, Bankauszüge |
| Secrets | `.env`, `backend/jwt_private.pem`, `backend/jwt_public.pem` | Fernet-Schlüssel (`ENCRYPTION_KEYS`), JWT-Schlüsselpaar, DB-Passwörter |
| Redis | Volume `redisdata` | Nur flüchtig (Sessions-Lockout, Celery-Queue) — **kein Backup nötig** |
| Ollama-Modelle | Volume `ollamadata` | Reproduzierbar via `ollama pull` — Backup optional |

> **Kritisch:** Ohne `ENCRYPTION_KEYS` aus der `.env` sind alle feldverschlüsselten
> PII und alle `.enc`-Dateien in `storage/` **unwiederbringlich verloren** —
> ein DB-Backup allein genügt nicht. Secrets getrennt und verschlüsselt sichern
> (z. B. Passwort-Manager der Kanzlei + Offline-Kopie im Safe).

## 2. Backup

### 2.1 Tägliches automatisches Backup (empfohlen)

```bash
#!/usr/bin/env bash
# /usr/local/bin/legal-backup.sh — per cron/systemd-timer täglich ausführen
set -euo pipefail
STAMP=$(date +%F_%H%M)
DEST=/backup/legal-automation/$STAMP     # z. B. gemountetes NAS/Backup-Volume
mkdir -p "$DEST"

cd /pfad/zu/legal-automation

# 1) Konsistenter DB-Dump (Custom-Format → selektives Restore möglich)
docker compose exec -T postgres pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  --format=custom > "$DEST/db.dump"

# 2) Dateispeicher (Inhalte sind bereits Fernet-verschlüsselt)
tar -czf "$DEST/storage.tar.gz" storage/

# 3) Secrets (nur wenn das Backup-Ziel selbst verschlüsselt ist!)
tar -czf "$DEST/secrets.tar.gz" .env backend/jwt_private.pem backend/jwt_public.pem

# 4) Aufbewahrung: 14 Tage rollierend (an Kanzlei-Löschkonzept anpassen)
find /backup/legal-automation -maxdepth 1 -mtime +14 -exec rm -rf {} +
```

- `POSTGRES_USER`/`POSTGRES_DB` aus der `.env` übernehmen (oder Skript sourcen).
- Backup-Ziel **verschlüsseln** (LUKS-Volume oder `age`/`gpg` auf die Archive).
- DSGVO: Backup-Aufbewahrung im Verarbeitungsverzeichnis dokumentieren;
  Art.-17-Löschungen erledigen sich in rollierenden Backups nach Ablauf des
  Aufbewahrungsfensters von selbst — Fenster daher klein halten.

### 2.2 Restore

```bash
cd /pfad/zu/legal-automation
docker compose down
# Secrets + storage zurückspielen
tar -xzf /backup/.../secrets.tar.gz
tar -xzf /backup/.../storage.tar.gz

# Datenbank neu aufbauen
docker compose up -d postgres
docker compose exec -T postgres dropdb  -U "$POSTGRES_USER" --if-exists "$POSTGRES_DB"
docker compose exec -T postgres createdb -U "$POSTGRES_USER" "$POSTGRES_DB"
docker compose exec -T postgres pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  --no-owner < /backup/.../db.dump

docker compose up -d       # Backend prüft/ergänzt Migrationen beim Start
curl -k https://localhost/api/health   # -> {"status":"ok", ...}
```

**Restore-Probe vierteljährlich durchführen** (auf Zweitsystem/VM): Ein Backup,
das nie zurückgespielt wurde, ist kein Backup. Prüfpunkte: Login, eine Akte
öffnen, ein Transkript-Original entschlüsselt herunterladen (beweist, dass
DB **und** `ENCRYPTION_KEYS` **und** `storage/` zusammenpassen).

## 3. Update-Prozess

```bash
cd /pfad/zu/legal-automation
/usr/local/bin/legal-backup.sh                    # 1) IMMER erst Backup

git fetch && git checkout <neuer-release-tag>     # 2) Neue Version holen

docker compose build backend frontend worker worker-beat   # 3) Images bauen
docker compose up -d                              # 4) Rollout
                                                  #    (Migrationen laufen beim
                                                  #    Backend-Start, Advisory-
                                                  #    Lock-serialisiert)
curl -k https://localhost/api/health              # 5) Smoke-Test
docker compose logs --since 5m backend worker-beat | grep -i error  # 6) Log-Check
```

- **Release-Notes lesen**: Migrationsschritte mit Handlungsbedarf (z. B. neue
  `.env`-Variablen) stehen dort. `diff .env.example .env` nach jedem Update.
- **Rollback**: `git checkout <alter-tag> && docker compose up -d --build`.
  Migrationen sind vorwärtsgerichtet — bei Schema-Änderungen zurück nur über
  das Backup aus Schritt 1 (deshalb ist Schritt 1 nicht optional).
- Wartungsfenster: Updates außerhalb der Kanzleizeiten; `docker compose up -d`
  ersetzt Container in Sekunden, laufende Transkriptions-Jobs werden von
  Celery nach Neustart erneut angestoßen (acks_late).

## 4. Festplattenverschlüsselung (LUKS)

`storage/`-Inhalte sind bereits anwendungsseitig verschlüsselt (Fernet); die
Datenbank (`pgdata`-Volume) ist es **nicht**. Für Diebstahl-/Entsorgungsschutz
des Servers gilt daher: **Full-Disk- oder Volume-Verschlüsselung ist Pflicht**
(TOM.md § „Verschlüsselung ruhender Daten").

Empfohlen: eigenes LUKS-Volume für Docker-Daten + storage:

```bash
cryptsetup luksFormat /dev/sdX                    # einmalig (VERNICHTET Daten!)
cryptsetup open /dev/sdX legaldata
mkfs.ext4 /dev/mapper/legaldata
mount /dev/mapper/legaldata /var/lib/docker       # oder dediziertes Datenverz.
```

- Passphrase im Kanzlei-Safe hinterlegen; optional Keyfile auf USB-Token.
- Nach jedem Server-Neustart muss das Volume manuell entsperrt werden
  (bewusste Entscheidung: kein Klartext-Keyfile auf derselben Platte).
- Alternative bei Hosting im RZ: verschlüsselte Volumes des Anbieters nur,
  wenn der Schlüssel bei der Kanzlei liegt.

## 5. Schlüssel-Rotation

### Fernet (`ENCRYPTION_KEYS`, MultiFernet)

1. Neuen Schlüssel erzeugen:
   `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
2. In `.env` **vorn anstellen**: `ENCRYPTION_KEYS=<neu>,<alt>` — Entschlüsselung
   probiert alle Schlüssel, Verschlüsselung nutzt den ersten.
3. `docker compose up -d` (Neustart Backend + Worker).
4. Alt-Schlüssel erst entfernen, wenn Altbestände umgeschlüsselt sind
   (Re-Encrypt-Lauf: Feld lesen → speichern genügt; für `.enc`-Dateien
   decrypt/encrypt-Skript — bis dahin beide Schlüssel behalten).

### JWT-Schlüsselpaar (RS256)

```bash
openssl genrsa -out backend/jwt_private.pem 4096
openssl rsa -in backend/jwt_private.pem -pubout -out backend/jwt_public.pem
docker compose up -d backend worker worker-beat
```

Folge: alle laufenden Sessions werden ungültig — Nutzer melden sich neu an.
Rotation bei Verdacht auf Kompromittierung sofort, sonst jährlich.

## 6. Monitoring & Logs

| Prüfung | Wie | Erwartung |
|---|---|---|
| Gesamtstatus | `curl -k https://<host>/api/health` | `{"status":"ok","db":"connected","redis":"connected"}` |
| Container | `docker compose ps` | 7 Services `running` (8 mit `--profile ki`) |
| Beat-Jobs laufen | `docker compose logs --since 10m worker-beat` | Sync-/SLA-Task-Zeilen, keine Tracebacks |
| Plattenplatz | `df -h` auf storage-/Docker-Volume | < 80 % belegt (Transkript-Audio wächst!) |
| Fehlversuche/Sperren | DSGVO-Admin-Dashboard im Frontend | keine ungewöhnlichen Häufungen |

- `docker compose logs -f backend` für Live-Fehlersuche; Log-Level über
  `LOG_LEVEL` in `.env`.
- health-Endpunkt eignet sich für externes Monitoring (Uptime-Kuma o. ä. —
  self-hosted, DSGVO-konform); Alarm bei ≠ 200 oder `"degraded"`.

## 7. Notfall-Checkliste (Desaster Recovery)

1. **Serverausfall:** Ersatzmaschine mit Docker → Repo klonen →
   Secrets + `storage/` + DB-Dump aus Backup einspielen (Abschnitt 2.2) →
   DNS/Hosts umstellen. Ziel-Wiederanlaufzeit mit der Kanzlei vereinbaren.
2. **Verschlüsselungs-Schlüssel verloren:** verschlüsselte Felder/Dateien sind
   ohne `ENCRYPTION_KEYS` nicht wiederherstellbar → deshalb Secrets-Backup an
   zwei getrennten Orten (Abschnitt 1).
3. **Kompromittierungsverdacht:** betroffene Nutzer deaktivieren (Admin-UI),
   JWT-Keys rotieren (alle Sessions tot), Passwort-Resets erzwingen,
   `audit_log` sichern und auswerten (unveränderlich, INSERT-only),
   Meldepflichten nach Art. 33/34 DSGVO prüfen (72 h!).
4. **Fehlgeschlagenes Update:** Container auf alten Tag zurückbauen; bei
   Schema-Bruch Restore aus dem Vor-Update-Backup (Abschnitt 3, Schritt 1).

---

*Dieses Handbuch beschreibt den technischen Betrieb. Aufbewahrungsfristen,
Löschkonzept und Verantwortlichkeiten regelt die Kanzlei in ihrem
Verarbeitungsverzeichnis (Art. 30 DSGVO) — Template unter `docs/dsgvo/`.*
