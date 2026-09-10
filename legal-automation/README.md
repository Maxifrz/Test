# Legal Automation Platform

DSGVO-konforme, **vollständig selbst gehostete** Automatisierungsplattform für
Anwaltskanzleien (Schwerpunkt Insolvenz-/Sanierungsrecht). Keine Cloud-Dienste,
keine Datenabflüsse — alle Verarbeitung (inkl. KI-Transkription, OCR und
Rechtsrecherche) läuft lokal.

## Module

| Modul | Kern |
|---|---|
| Auth & RBAC | JWT (RS256), TOTP-2FA (für Anwalt/Admin **erzwungen**, kein Bypass über Pflicht-Passwortwechsel), Rollen + aktenbezogene Rechte (`matter_access`), unveränderliches Audit-Log |
| **Benutzerverwaltung** | Anlage mit Einmal-Passwort, Rollenwechsel, Passwort-/2FA-Reset, Deaktivierung; der letzte aktive Admin ist geschützt |
| Mandanten & Akten | Stammdaten mit feldverschlüsselten Kontaktdaten + Blind-Index, Aktenzugriff pro Nutzer, Aufbewahrungsfristen |
| E-Mail | **Mehrere Postfächer je Standort**, IMAP-Ingest + Regel-Zuordnung, Review-Queue, Ladungs-Parser, Vorlagen (sandboxed), **Outbox mit Retry**, BCC bei Mehrfachempfängern |
| **Dokumente** | Verschlüsselte Ablage, Textextraktion aus PDF, **lokales OCR** (Tesseract) für Scans, deutsche Volltextsuche |
| Aufgaben & Fristen | Tickets, SLA, **Fristen-Kalkulator**: ZPO/StPO **und InsO** (§§ 28, 146, 178/179, 189, 113 InsO, § 324 SGB III), Feiertage je Bundesland, § 193 BGB |
| Kalender | Termine + Konflikt-/Feiertagsprüfung, Gerichtstermin → Vorbereitungs-Tickets, ICS-Import/-Export |
| Transkription | faster-whisper + pyannote **lokal**, eigener Worker, editierbare Segmente mit Edit-Historie, deutsche Volltextsuche |
| Finanzen (Insolvenz) | Massekonten, CAMT.053/MT940-Import, **InsVV-Rechner (Fassung seit 2021)**, **RVG mit Anrechnung nach Vorbem. 3 Abs. 4 VV**, Forderungstabelle, **Verteilungsrechner nach InsO-Rangfolge**, Vergütungsantrag- und **Tabellenauszug-PDF (§ 178 III InsO)** |
| Gläubiger-Portal | Öffentliche, token-gesicherte Online-Forderungsanmeldung (`portal/`), strikt limitiert |
| DSGVO-Tools | Verarbeitungsverzeichnis (Art. 30), **Löschregistry über 17 Tabellen** (Art. 17), Datenexport (Art. 20), Admin-Dashboard |
| KI-Recherche | GraphRAG: Wissensgraph (pgvector + Zitationskanten + FTS), Hybrid-Retrieval **mit Reranking**, **inhaltliche Grounding-Prüfung**, Korpus-Versionierung. Opt-in: `KI_ENABLED=true` + `docker compose --profile ki up` |
| **Betrieb** | `/api/metrics` (Prometheus), Liveness-/Readiness-Endpunkte, automatisierte Restore-Probe |

## Quickstart

> Postgres läuft als `pgvector/pgvector:pg16` (Drop-in-Ersatz für postgres:16
> mit vector-Extension — Voraussetzung der Migration 0010).

Voraussetzungen: Docker + Docker Compose, `openssl`.

```bash
./setup.sh          # erzeugt .env, Secrets (inkl. Redis-Passwort), JWT-Keys, SSL-Zertifikat
docker compose up -d
# → https://<host>/  — Login: admin@kanzlei.local / Admin1234!
```

Beim ersten Login werden **Passwortwechsel und 2FA-Einrichtung erzwungen**.
Migrationen laufen automatisch beim Backend-Start (Advisory-Lock-geschützt).

Die Secrets liegen unter `secrets/` — **außerhalb** der Docker-Build-Kontexte,
damit `COPY . .` den privaten JWT-Schlüssel nicht in ein Image backen kann.

Tests:

```bash
cd backend
python -m pytest tests/ -q --ignore=tests/integration   # 300 Unit-Tests (reine Logik)
RUN_INTEGRATION=1 python -m pytest tests/integration/ -q # API-Flows gegen echtes Postgres+Redis
ruff check .                                             # Lint
cd ../frontend && npm install && npm run build           # tsc + vite
```

CI (`.github/workflows/ci.yml`) fährt zusätzlich einen **Compose-Smoketest**:
den ganzen Stack hochfahren, `/api/health` prüfen, die Login-Kette durchspielen
und verifizieren, dass kein Schlüssel im Image liegt. Unit- und
Integrationstests umgehen den Compose-Stack — genau dort saßen die Fehler, die
den Start verhindert haben.

## Architektur

```
nginx (TLS 1.3, Rate-Limits, CSP)  ──► backend   FastAPI (88 Endpunkte, /api)
                                   ──► frontend  React/TS (Vite → nginx)
                                   ──► portal/   statisches Gläubiger-Portal
                                   ──► website/  Kanzlei-Website (eigener vHost)
backend ──► postgres 16 + pgvector (FTS, Audit-Trigger, RLS-Policies)
        ──► redis (Lockout, Celery-Broker)

worker              E-Mail-Sync/-Versand, SLA, Retention, KI-Ingestion
worker-transcription  Transkription + OCR (eigener Worker, damit ein
                      Whisper-Lauf den E-Mail-Sync nicht aushungert)
worker-beat         Zeitpläne (crontab, nicht Intervalle ab Beat-Start)
```

Muster im Code: rechtlich kritische Logik ist **rein und getestet**
(`services/deadline_calculator.py`, `rvg_calculator.py`, `insvv_calculator.py`,
`distribution.py`, `dsgvo_retention.py`, `dsgvo_erasure.py`, `ai/kri/*`);
ML-Abhängigkeiten werden lazy importiert; Geldbeträge sind durchgängig
`Decimal`.

## Sicherheit / DSGVO (Kurzfassung, Details: `docs/dsgvo/TOM.md`)

- **Verschlüsselt:** Mandanten-E-Mail (+ Blind-Index für die exakte Suche),
  Telefon, Anschrift, Geburtsdatum, Steuer-ID, Notizen; E-Mail-Anhänge,
  Dokumente und Original-Audio auf der Platte.
- **Bewusst nicht feldverschlüsselt:** Namen und Ort (Namenssuche und
  Sortierung sind der zentrale Arbeitsablauf) sowie Akteninhalte (die deutsche
  Volltextsuche braucht lesbaren Text). Der Schutz ruht dort auf LUKS, RBAC und
  Aktentrennung. Die Abwägung steht ausformuliert in der TOM — die Doku
  behauptet nicht mehr, als der Code hält.
- Aktentrennung wird **anwendungsseitig auf jedem Endpunkt** durchgesetzt.
  Zusätzlich liegen **Postgres-RLS-Policies** bereit (Migration 0020), die der
  Betreiber mit `SELECT enable_matter_rls();` scharf schaltet — zweite
  Verteidigungslinie gegen einen vergessenen Zugriffscheck.
- Audit-Log ist per DB-Trigger unveränderlich; Schreibvorgänge werden bis zum
  Shutdown abgewartet und Fehlschläge protokolliert (kein stiller Verlust).
- `backend/init_db.sql` legt einen echten Least-Privilege-Laufzeit-User an, der
  `audit_log` nur beschreiben darf (opt-in über `APP_RUNTIME_PASSWORD`).
- Öffentliches Portal: unguessbares Token pro Verfahren, nur Anlegen, strenge
  nginx-Rate-Limits, Datensparsamkeit (nur Aktenzeichen sichtbar).

## Bekannte Limitierungen (Stand jetzt)

- **Kein beA-Anschluss.** Das ist die größte verbleibende Lücke: für eine
  deutsche Kanzlei ist beA faktisch Pflicht. Eine echte Anbindung erfordert
  OSCI-Transport, ein SAFE-Verzeichnis, kartengebundene Zertifikate und eine
  Zulassung — das lässt sich nicht seriös nebenbei implementieren und ist
  bewusst **nicht** angefangen worden, statt eine Attrappe zu bauen. Der
  Dokumenten-Import ist so gebaut, dass eine spätere beA-Quelle daran andockt.
- **OCR-Qualität** hängt an der Scan-Qualität. `documents.text_from_ocr`
  markiert Texte, die per OCR entstanden sind — sie sind fehleranfällig und
  taugen für die Suche, nicht als Zitatgrundlage.
- **E-Mail-Anhänge > 25 MB** werden nicht extrahiert (bleiben im Postfach).
- **Absenderprüfung**: die Zuordnung eingehender Mail vertraut dem
  `From:`-Header; SPF/DKIM werden nicht ausgewertet.
- Access-Token liegt im `localStorage` (CSP mildert XSS-Risiko).
- Tickets ohne Aktenbezug sind rollenweit sichtbar (bewusste Entscheidung).
- **KI-Quellen-Adapter** sind gegen Format-Fixtures getestet; vor dem ersten
  Produktiv-Ingest einen Live-Download verifizieren (Format-Drift).
- Die **inhaltliche Grounding-Prüfung** ist ein erklärbares Überlappungsmaß,
  kein NLI-Modell: sie fängt frei erfundene Passagen ab, ersetzt aber nicht die
  anwaltliche Prüfung. Jede KI-Ausgabe bleibt ein Entwurf (RDG).
- `ruff format` ist eingerichtet, aber **kein CI-Gate** — der Formatter würde
  112 Dateien in einem Rutsch umschreiben.

## Go-Live-Checkliste

1. `docker compose up` → alle Container laufen (insb. `worker-beat` und
   `worker-transcription`). Der Compose-Smoketest in CI deckt das ab.
2. Login-Kette: Passwortwechsel → 2FA-Setup → Dashboard.
3. **Benutzer anlegen** (`/benutzer`) und Aktenrechte vergeben — ohne das
   arbeitet die Kanzlei mit einem einzigen Admin-Konto.
4. **Postfächer anlegen** (`POST /api/email-accounts`) und je Konto den
   Verbindungstest laufen lassen.
5. Transkriptions- und Dokumenten-Upload testen (nginx-Locations).
6. Nutzer ohne `matter_access` → 403 auf fremde Akte/E-Mail/Konto.
7. Website-vHost: echte Domain + Zertifikat in `nginx/nginx.conf` eintragen
   (⟨Platzhalter⟩), Kontaktformular-Testversand.
8. **Least-Privilege-DB-User + RLS aktivieren** (`docs/OPERATIONS.md`, 5a).
9. Backup einrichten und **Restore-Probe als Timer** einrichten
   (`scripts/restore-probe.sh`).
10. Monitoring auf `legal_email_outbox_failed` und die Queue-Längen legen
    (`docs/OPERATIONS.md`, 6a).
11. **Anwaltliche Freigabe der Rechtswerte** — im Code als „vor Go-Live prüfen"
    markiert und bewusst an einer Stelle je Modul gebündelt:
    - `services/insvv_calculator.py`: `_STAFFEL` und die
      `MINDESTVERGUETUNG_*`-Konstanten (Fassung seit 2021 hinterlegt)
    - `services/rvg_calculator.py`: `_TIERS`, `_OVER_500K`, `ANRECHNUNG_*`
    - `services/deadline_calculator.py`: die benannten Fristen, insbesondere
      die **gerichtlich gesetzte** Anmeldefrist nach § 28 InsO
    - `services/dsgvo_retention.py`: `BRAO_RETENTION_YEARS`, `AO_RETENTION_YEARS`
    - `services/dsgvo_erasure.py`: die Löschregistry (welche Felder gelöscht,
      welche aus Rechtsgründen behalten werden)
