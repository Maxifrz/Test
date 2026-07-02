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
| KI (V3.0, in Arbeit) | GraphRAG-Rechtsrecherche: struktur-bewusstes Chunking, Zitations-Graph, Hybrid-Retrieval, **grounded-only** Antworten (Kernlogik fertig + getestet) |

## Quickstart

Voraussetzungen: Docker + Docker Compose, `openssl`.

```bash
./setup.sh          # erzeugt .env, Secrets, JWT-Keys, SSL-Zertifikat
docker compose up -d
# → https://<host>/  — Login: admin@kanzlei.local / Admin1234!
```

Beim ersten Login werden **Passwortwechsel und 2FA-Einrichtung erzwungen**.
Migrationen laufen automatisch beim Backend-Start (Advisory-Lock-geschützt).

Tests (Backend, reine Logik — kein Docker nötig):

```bash
cd backend && python -m pytest tests/ -q     # 125 Tests
cd frontend && npm install && npm run build  # tsc + vite
```

## Architektur

```
nginx (TLS 1.3, Rate-Limits, CSP) ──► backend  FastAPI (59 Endpunkte, /api)
                                  ──► frontend React/TS (Vite)
                                  ──► portal/  statisches Gläubiger-Portal
backend ──► postgres 16 (30 Tabellen, FTS, Audit-Trigger)
        ──► redis (Sessions-Lockout, Celery-Broker)
worker / worker-beat (Celery): E-Mail-Sync, Transkription, SLA, Retention-Report
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

- **E-Mail-Anhänge** werden noch nicht extrahiert/abgelegt (Tabelle existiert;
  Anhänge bleiben im IMAP-Postfach). Geplant: verschlüsselte Ablage.
- **Ein Postfach**: IMAP/SMTP global per `.env` — Multi-Postfach (z. B. je
  Standort) erfordert eine `email_accounts`-Erweiterung.
- **Audit-Middleware** schreibt asynchron fire-and-forget; bei hartem
  Prozess-Stop können einzelne Einträge verloren gehen.
- Access-Token liegt im `localStorage` (CSP mildert XSS-Risiko).
- Tickets ohne Aktenbezug sind rollenweit sichtbar (bewusste Entscheidung).
- **Integrationstests** (API-Ebene) fehlen noch — Unit-Abdeckung der Kernlogik
  ist hoch (125 Tests), Live-Smoke-Test siehe unten.

## Go-Live-Checkliste

1. `docker compose up` → alle 7 Container laufen (insb. `worker-beat`).
2. Login-Kette: Passwortwechsel → 2FA-Setup → Dashboard.
3. Transkriptions-Upload > 1 MB (nginx-Location `/api/transcriptions`).
4. `GET /api/emails/templates` → 200.
5. Nutzer ohne `matter_access` → 403 auf fremde Akte/E-Mail/Konto.
6. **Anwaltliche Freigabe** der Rechtswerte: Fristenregeln, RVG-Tabelle,
   InsVV-Sätze/Mindestvergütung, DSGVO-Texte (im Code als
   „vor Go-Live prüfen" markiert).
