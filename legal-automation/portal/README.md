# Öffentliches Gläubiger-Portal

Selbstständige, statische Web-App für die **Online-Forderungsanmeldung** durch
Gläubiger. Kein Build-Schritt, keine externen Abhängigkeiten (CSP-konform:
externes JS aus gleicher Herkunft, Inline-Styles).

## Funktionsweise

- Zugang über verfahrensspezifischen Link: `https://<host>/portal/?token=<TOKEN>`
- Das Token wird im internen Tool pro Akte erzeugt
  (`POST /api/insolvency/matters/{id}/creditor-portal`, Rolle `finance.write`).
- Die App spricht **ausschließlich** die öffentlichen Endpunkte an:
  - `GET  /api/public/creditor-claims/{token}` → Verfahren prüfen/anzeigen
  - `POST /api/public/creditor-claims/{token}` → Forderung anmelden
- Angemeldete Forderungen erscheinen im internen Tool (Quelle „Portal", Status
  „angemeldet") und werden dort geprüft/festgestellt.

## Sicherheit / DSGVO

- Unauthentifiziert, aber token-gesichert; nur **Anlegen**, kein Lese-/Änderungs-
  zugriff auf bestehende Daten.
- Strenge Ratenbegrenzung in nginx (`zone=portal`, 10 r/min, burst 5).
- Inhalte gelten als nicht vertrauenswürdig (Untrusted) und werden serverseitig
  erst durch Sachbearbeiter geprüft.
- `noindex, nofollow`.

## Deployment

Im Standard-Setup liefert der vorhandene nginx das Portal unter `/portal/`
(Volume `./portal:/usr/share/nginx/portal:ro`).

**Empfohlene Härtung (Produktion):** Portal auf einem **eigenen vHost / eigener
(Sub-)Domain** ausliefern, der nur `/portal/` und `/api/public/` exponiert und
vom internen Kanzlei-Netz getrennt ist. Dazu einen zweiten `server`-Block bzw.
eine zweite nginx-Instanz verwenden, die ausschließlich diese beiden Locations
kennt — der interne `/api/`- und SPA-Zugang bleibt dann komplett unerreichbar.
