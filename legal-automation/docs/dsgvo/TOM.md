# Technisch-Organisatorische Maßnahmen (TOM)
## gem. Art. 32 DSGVO

**Produkt**: Legal Automation Platform  
**Version**: 0.1.0  
**Stand**: 2025-01

---

## 1. Zugangskontrolle

- TLS 1.3 für alle Verbindungen (nginx); ältere Protokolle deaktiviert
- Passwörter mit bcrypt (Cost Factor 12) gehasht
- Zwei-Faktor-Authentifizierung (TOTP) Pflicht für Anwälte und Admins
- Automatische Kontosperrung nach 5 fehlgeschlagenen Anmeldeversuchen
- Automatischer Session-Timeout nach 30 Minuten Inaktivität
- Brute-Force-Schutz via Rate-Limiting (nginx + Redis)

## 2. Zugriffskontrolle

- Rollenbasiertes Zugriffsmodell (RBAC): Admin, Anwalt, Sachbearbeiter, Sekretariat
- Akten-spezifische Zugriffsberechtigungen (matter_access)
- Kein Nutzer kann auf Akten zugreifen, für die keine explizite Berechtigung erteilt wurde
- Datenbankebene: Row-Level-Security zusätzlich zu Anwendungslogik

## 3. Weitergabekontrolle

- Keine Datenübermittlung an Drittanbieter oder Cloud-Dienste
- Alle Verarbeitungen erfolgen auf dem lokalen Server
- Whisper-Transkription läuft lokal, keine Audiodaten verlassen den Server
- API-Keys und Geheimnisse nur über Umgebungsvariablen, nie im Code

## 4. Eingabekontrolle (Audit-Log)

- Jede Aktion (Lesen, Erstellen, Ändern, Löschen) wird im audit_log protokolliert
- audit_log ist unveränderlich: Datenbankbenutzer hat nur INSERT-Rechte
- Trigger verhindert UPDATE und DELETE auf audit_log-Ebene
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

- PII-Felder (Geburtsdatum, Steuer-ID, Notizen) mit Fernet (AES-128-CBC) verschlüsselt
- Schlüssel-Rotation unterstützt (MultiFernet)
- Transkriptions-Audiodateien nach Verarbeitung verschlüsselt gespeichert (original.enc)
- LUKS-Festplattenverschlüsselung für den Storage-Ordner empfohlen (Betreiber-Verantwortung)

## 8. Datensparsamkeit

- Intermediate-Dateien (WAV für Transkription) werden nach Verarbeitung sofort gelöscht
- Passwörter und Secrets werden nie im Klartext gespeichert
- Audit-Log speichert anonymisierte Details bei Löschanträgen
