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

- PII-Felder (Geburtsdatum, Steuer-ID, Notizen) mit Fernet (AES-128-CBC) verschlüsselt
- Schlüssel-Rotation unterstützt (MultiFernet)
- Transkriptions-Audiodateien nach Verarbeitung verschlüsselt gespeichert (original.enc)
- LUKS-Festplattenverschlüsselung für den Storage-Ordner empfohlen (Betreiber-Verantwortung)

## 8. Datensparsamkeit

- Intermediate-Dateien (WAV für Transkription) werden nach Verarbeitung sofort gelöscht
- Passwörter und Secrets werden nie im Klartext gespeichert
- Audit-Log-Einträge bleiben bei Löschungen (Art. 17) aus Integritätsgründen erhalten;
  der Personenbezug entfällt durch Anonymisierung der referenzierten Stammdaten —
  die Einträge selbst enthalten keine Klartext-Personendaten (nur Aktion/Ressource/Ergebnis)
