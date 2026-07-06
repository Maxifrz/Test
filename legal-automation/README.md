# Legal Automation Platform

DSGVO-konforme, **vollständig selbst gehostete** Automatisierungsplattform für
Anwaltskanzleien (Schwerpunkt Insolvenz-/Sanierungsrecht). Keine Cloud-Dienste,
keine Datenabflüsse — alle Verarbeitung (inkl. KI-Transkription) läuft lokal.

## Module (MVP, Phase 0–7)

| Modul | Kern |
|---|---|
| Auth & RBAC | JWT (RS256), TOTP-2FA (für Anwalt/Admin **erzwungen**), Rollen + aktenbezogene Rechte (`matter_access`), unveränderliches Audit-Log |
| Mandanten & Akten | Stammdaten (PII feldverschlüsselt, Fernet), Aktenzugriff pro Nutzer, Aufbewahrungsfristen |
| E-Mail | IMAP-Ingest + Regel-Zuordnung zu Akten, Review-Queue, Ladungs-Parser (Termin/Az/Saal), Vorlagen, SMTP-Versand |
| Aufgaben & Fristen | Tickets, SLA, **Fristen-Kalkulator** (ZPO/BGB-Regeln, Feiertage je Bundesland, § 193 BGB) |
| Kalender | Termine + Konflikt-/Feiertagsprüfung, Gerichtstermin → automatische Vorbereitungs-Tickets, ICS-Import/-Export |
| Transkription | faster-whisper + pyannote **lokal**, Celery-Pipeline, editierbare Segmente mit Edit-Historie, deutsche Volltextsuche |
| Finanzen (Insolvenz) | Massekonten, CAMT.053/MT940-Import mit Saldo-Abgleich, **InsVV-/RVG-Rechner**, Forderungstabelle, **Verteilungsrechner** (quotal, cent-genau), Vergütungsantrag-PDF |
| Gläubiger-Portal | Öffentliche, token-gesicherte Online-Forderungsanmeldung (`portal/`), strikt limitiert |
| DSGVO-Tools | Verarbeitungsverzeichnis (Art. 30), geprüfter Löschworkflow (Art. 17, blockiert bei § 50 BRAO/§ 147 AO), Datenexport (Art. 20, 48 h/single-use), Admin-Dashboard |
| KI-Recherche (V3.0) | GraphRAG: Wissensgraph (pgvector + Zitationskanten + FTS), Hybrid-Retrieval, **grounded-only** Antworten mit [S#]-Belegen (Halluzination → Ablehnung), Audit + Feedback. Opt-in: `KI_ENABLED=true` + `docker compose --profile ki up` + `ollama pull` |

## Quickstart

> Hinweis: Postgres läuft als `pgvector/pgvector:pg16` (Drop-in-Ersatz für
> postgres:16 mit vector-Extension — Voraussetzung der Migration 0010).


Voraussetzungen: Docker + Docker Compose, `openssl`.

```bash
./setup.sh          # erzeugt .env, Secrets, JWT-Keys, SSL-Zertifikat
docker compose up -d
# → https://<host>/  — Login: admin@kanzlei.local / Admin1234!
```

Beim ersten Login werden **Passwortwechsel und 2FA-Einrichtung erzwungen**.
Migrationen laufen automatisch beim Backend-Start (Advisory-Lock-geschützt).

Tests:

```bash
cd backend && python -m pytest tests/ -q --ignore=tests/integration  # 145 Unit-Tests (reine Logik)
RUN_INTEGRATION=1 python -m pytest tests/integration/ -q  # 12 API-Flows gegen echtes Postgres+Redis
cd frontend && npm install && npm run build  # tsc + vite
```

Die Integrationstests bauen das Schema pro Lauf komplett neu auf
(`alembic upgrade head`) und testen damit zugleich die Migrationskette;
sie laufen automatisch in CI (`.github/workflows/ci.yml`, pgvector- und
redis-Service-Container). **Betrieb** (Backup/Restore, Updates, LUKS,
Schlüssel-Rotation, Notfall): `docs/OPERATIONS.md`.

## Architektur

```
nginx (TLS 1.3, Rate-Limits, CSP) ──► backend  FastAPI (72 Endpunkte, /api)
                                  ──► frontend React/TS (Vite)
                                  ──► portal/  statisches Gläubiger-Portal
                                  ──► website/ Kanzlei-Website (eigener vHost,
                                      Kontaktformular → /api/public/contact)
backend ──► postgres 16 + pgvector (36 Tabellen, FTS, Audit-Trigger)
        ──► redis (Sessions-Lockout, Celery-Broker)
worker / worker-beat (Celery): E-Mail-Sync, Transkription, SLA, Retention-Report,
                               KI-Bulk-Ingestion (gesetze-im-internet / RII)
```

Muster im Code: rechtlich kritische Logik ist **rein und getestet**
(`services/frist_calculator.py`, `insvv_calculator.py`, `distribution.py`,
`ai/kri/*`); ML-Abhängigkeiten werden lazy importiert; Geldbeträge sind
durchgängig `Decimal`.

## Sicherheit / DSGVO (Kurzfassung, Details: `docs/dsgvo/TOM.md`)

- Aktentrennung wird **anwendungsseitig auf jedem Endpunkt** durchgesetzt
  (`ensure_matter_access` + Listen-Filter). Keine DB-RLS (bewusst dokumentiert).
- Audit-Log ist per DB-Trigger unveränderlich. `backend/init_db.sql` stellt
  zusätzlich einen Least-Privilege-DB-User bereit — **wirksam nur, wenn der
  Betreiber ihn als Laufzeit-User konfiguriert**.
- Öffentliches Portal: unguessbares Token pro Verfahren, nur Anlegen, strenge
  nginx-Rate-Limits, Datensparsamkeit (nur Aktenzeichen sichtbar).

## Bekannte Limitierungen (Stand jetzt)

- **E-Mail-Anhänge** > 25 MB werden nicht extrahiert (bleiben im IMAP-Postfach);
  kleinere werden Fernet-verschlüsselt unter `storage/emails/` abgelegt.
- **Ein Postfach**: IMAP/SMTP global per `.env` — Multi-Postfach (z. B. je
  Standort) erfordert eine `email_accounts`-Erweiterung.
- **Audit-Middleware** schreibt asynchron fire-and-forget; bei hartem
  Prozess-Stop können einzelne Einträge verloren gehen.
- Access-Token liegt im `localStorage` (CSP mildert XSS-Risiko).
- Tickets ohne Aktenbezug sind rollenweit sichtbar (bewusste Entscheidung).
- **KI-Quellen-Adapter** sind gegen Format-Fixtures getestet; vor dem ersten
  Produktiv-Ingest einen Live-Download verifizieren (Format-Drift).

## Go-Live-Checkliste

1. `docker compose up` → alle 7 Container laufen (insb. `worker-beat`).
2. Login-Kette: Passwortwechsel → 2FA-Setup → Dashboard.
   (Kette + Migrationen sind durch `tests/integration/` abgedeckt und in CI
   grün — Punkt 2–5 sind dort automatisiert, hier nur Sichtprüfung.)
3. Transkriptions-Upload > 1 MB (nginx-Location `/api/transcriptions`).
4. `GET /api/emails/templates` → 200.
5. Nutzer ohne `matter_access` → 403 auf fremde Akte/E-Mail/Konto.
6. Website-vHost: echte Domain + Zertifikat in `nginx/nginx.conf` eintragen
   (⟨Platzhalter⟩), Kontaktformular-Testversand → Eintrag unter
   „Kontaktanfragen" im Tool.
7. Backup einrichten und **Restore-Probe durchführen** (`docs/OPERATIONS.md`).
8. **Anwaltliche Freigabe** der Rechtswerte: Fristenregeln, RVG-Tabelle,
   InsVV-Sätze/Mindestvergütung, DSGVO-Texte (im Code als
   „vor Go-Live prüfen" markiert).
