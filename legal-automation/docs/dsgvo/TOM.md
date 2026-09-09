# Technisch-Organisatorische Maßnahmen (TOM)
## gem. Art. 32 DSGVO

**Produkt**: Legal Automation Platform  
**Version**: 0.1.0  
**Stand**: 2025-01

---

## 1. Zugangskontrolle

- TLS 1.3 für alle Verbindungen (nginx); ältere Protokolle deaktiviert
- Passwörter mit bcrypt (Cost Factor 12) gehasht
- Zwei-Faktor-Authentifizierung (TOTP) Pflicht für Anwälte und Admins — beim Login
  erzwungen: ohne eingerichtetes TOTP wird nur ein eingeschränkter Setup-Zugang
  ausgestellt, der ausschließlich die 2FA-Einrichtung erlaubt
- Initialer Admin-Account muss das Passwort beim ersten Login ändern (erzwungen)
- Automatische Kontosperrung nach 5 fehlgeschlagenen Anmeldeversuchen
- Automatischer Session-Timeout nach 30 Minuten Inaktivität
- Brute-Force-Schutz via Rate-Limiting (nginx + Redis)

## 2. Zugriffskontrolle

- Rollenbasiertes Zugriffsmodell (RBAC): Admin, Anwalt, Sachbearbeiter, Sekretariat
- Akten-spezifische Zugriffsberechtigungen (matter_access)
- Kein Nutzer kann auf Akten zugreifen, für die keine explizite Berechtigung erteilt
  wurde — durchgesetzt in der Anwendungsschicht auf jedem aktengebundenen Endpunkt
  (Detail-Guards `ensure_matter_access` + Listen-Filter über alle Module: Akten,
  E-Mails, Transkriptionen, Finanzen, Forderungen, Kalender, Tickets)
- Datenbankseitige Row-Level-Security ist NICHT implementiert (optionale spätere
  Härtung); maßgeblich ist die anwendungsseitige Durchsetzung

## 3. Weitergabekontrolle

- Keine Datenübermittlung an Drittanbieter oder Cloud-Dienste
- Alle Verarbeitungen erfolgen auf dem lokalen Server
- Whisper-Transkription läuft lokal, keine Audiodaten verlassen den Server
- API-Keys und Geheimnisse nur über Umgebungsvariablen, nie im Code

## 4. Eingabekontrolle (Audit-Log)

- Jede Aktion (Lesen, Erstellen, Ändern, Löschen) wird im audit_log protokolliert
- audit_log ist unveränderlich: ein Datenbank-Trigger verhindert UPDATE und DELETE
  (wirksam für alle DB-Nutzer)
- Zusätzlich stellt init_db.sql einen Least-Privilege-DB-User bereit (nur INSERT auf
  audit_log); wirksam, wenn der Betreiber ihn als Laufzeit-User konfiguriert
- Protokollierung enthält: Nutzer-ID, E-Mail, Session-ID, IP-Adresse, Zeitstempel, Aktion, Ergebnis

## 5. Verfügbarkeitskontrolle

- Docker Compose mit `restart: unless-stopped` für alle Services
- PostgreSQL mit persistentem Volume (pgdata)
- Redis mit persistentem Volume + AOF-Logging
- Backup-Strategie: Verantwortung des Betreibers (pg_dump empfohlen, täglich)

## 6. Trennungsgebot

- Mandantendaten werden durch matter_access-Berechtigungen getrennt
- Jeder Anwalt sieht nur eigene Akten (außer Admin)
- Transkriptionen sind an Akten gebunden und folgen denselben Zugriffsregeln

## 7. Verschlüsselung

### 7.1 Feldverschlüsselung (Fernet / AES-128-CBC + HMAC)

**Verschlüsselt in der Datenbank:**

| Tabelle | Felder |
|---|---|
| `clients` | E-Mail, Telefon, Adresszeilen 1+2, PLZ, Geburtsdatum, Steuer-ID, Notizen |

**Bewusst NICHT feldverschlüsselt — mit Begründung:**

| Felder | Begründung |
|---|---|
| `clients.first_name`, `last_name`, `company_name`, `city` | Namenssuche (Teilstring) und alphabetische Sortierung sind der zentrale Arbeitsablauf der Kanzlei. Verschlüsselt wären beide nur durch vollständiges Laden und Sortieren in der Anwendung möglich — das skaliert nicht und verlagert die Daten in den Anwendungsspeicher, ohne den Schutz zu erhöhen. |
| Akteninhalte (E-Mail-Texte, Transkripte, Aktenrubren) | Deutsche Volltextsuche (`tsvector`) und die KI-Recherche setzen lesbaren Text in der Datenbank voraus. |
| Verknüpfungen (Mandant ↔ Akte ↔ Termin) | Fremdschlüssel sind prinzipiell nicht verschlüsselbar; die Zuordnung selbst bleibt sichtbar. |

Der Schutz dieser Felder ruht damit auf: **Datenträgerverschlüsselung (LUKS)**,
**RBAC**, **Aktentrennung** und dem **Least-Privilege-DB-User** — nicht auf
Feldverschlüsselung. Das ist eine bewusste Abwägung und keine Lücke, aber sie
gehört ausgesprochen: die Feldverschlüsselung schützt gegen das Abfließen eines
Datenbank-Dumps, nicht gegen einen kompromittierten Anwendungsserver.

### 7.2 Blind-Index

Die verschlüsselte Mandanten-E-Mail ist über einen deterministischen
HMAC-SHA256 (`clients.email_index`, Schlüssel `PII_INDEX_KEY`) exakt
durchsuchbar — nötig für die automatische Zuordnung eingehender Nachrichten.
Grenzen: nur Gleichheitssuche, keine Teilstring-Suche. Wer HMAC-Schlüssel und
eine Kandidatenliste besitzt, kann Adressen bestätigen (bauartbedingt bei jedem
Blind-Index).

### 7.3 Weiteres

- Schlüssel-Rotation unterstützt (MultiFernet, mehrere Schlüssel kommagetrennt)
- E-Mail-Anhänge werden Fernet-verschlüsselt unter `storage/emails/` abgelegt
- Transkriptions-Audiodateien nach Verarbeitung verschlüsselt gespeichert (original.enc)
- LUKS-Festplattenverschlüsselung für den Storage-Ordner empfohlen (Betreiber-Verantwortung)
- Transport: TLS 1.3, HSTS, keine Klartext-Ports nach außen

## 8. Datensparsamkeit

- Intermediate-Dateien (WAV für Transkription) werden nach Verarbeitung sofort gelöscht
- Passwörter und Secrets werden nie im Klartext gespeichert
- Audit-Log-Einträge bleiben bei Löschungen (Art. 17) aus Integritätsgründen erhalten;
  die Einträge selbst enthalten keine Klartext-Inhaltsdaten (nur Aktion/Ressource/Ergebnis)

## 9. Löschung (Art. 17 DSGVO)

Die Löschung läuft über eine **deklarative Löschregistry**
(`backend/app/services/dsgvo_erasure.py`), nicht über handgeschriebenen Code je
Tabelle. Erfasst sind:

| Datenbestand | Maßnahme |
|---|---|
| Mandanten-Stammdaten | anonymisiert (inkl. Blind-Index) |
| Akten (Rubrum, Gegenseite, Beschreibung) | anonymisiert; Aktenzeichen, Fristen und Status bleiben (Aktenführung, § 50 BRAO) |
| E-Mails (Absender, Betreff, Inhalt, Empfängerlisten) | anonymisiert |
| E-Mail-Anhänge | Dateien auf der Platte werden gelöscht |
| Transkripte (Volltext, Segmente, Änderungshistorie, Originalaudio) | anonymisiert, Audio gelöscht |
| Forderungsanmeldungen (Gläubigerdaten) | anonymisiert; Beträge/Rang/Status bleiben (Verteilungsrechnung) |
| Termine und externe Teilnehmer | anonymisiert |
| Aufgaben und Fristen | Texte anonymisiert; Fristdaten bleiben (Nachweis der Berufspflichten) |
| KI-Recherchen (Frage, Entwurf, Feedback) | anonymisiert |
| Interne Dokumente im KI-Korpus | anonymisiert |
| Kontaktanfragen von der Website | anonymisiert (Zuordnung über die E-Mail-Adresse) |
| Audit-Log | **unverändert** — unveränderlich per DB-Trigger, Rechenschaftspflicht Art. 5 Abs. 2 |

Das ausgestellte **Löschzertifikat listet auf, was tatsächlich gelöscht wurde**
(Tabelle, Anzahl Zeilen, Anzahl Dateien) statt eine pauschale Aussage zu
treffen. Ein Abdeckungstest (`tests/test_dsgvo_erasure.py`) hält die Registry
gegen die Modelle: eine neue Tabelle mit Personenbezug lässt den Test fehlschlagen,
solange sie nicht eingetragen ist.

**Vorgelagerte Prüfung:** Eine Löschung ist blockiert, solange Akten offen sind
oder Aufbewahrungsfristen laufen. Die Fristen beginnen mit **Ablauf des
Kalenderjahres** (§ 50 Abs. 1 S. 2 BRAO, § 147 Abs. 4 AO, § 257 Abs. 5 HGB) —
nicht am Tagesdatum des Aktenschlusses. Als steuerrelevant markierte Akten
(`matters.tax_relevant`, bei Insolvenz-/Sanierungsakten vorbelegt) unterliegen
der 10-Jahres-Frist des § 147 Abs. 3 AO.
