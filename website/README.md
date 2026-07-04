# ECKERT — Marketing-Website

Statische Website der Kanzlei ECKERT (Design „Akte & Messing", aus dem
Claude-Design-Prototyp konvertiert und ausgebaut). Kein Build-Schritt —
direkt deploybar (eigener vHost, strikt getrennt vom internen Tool).

## Seiten
index · leistungen (+insolvenzverwaltung-Detail) · team · standorte
(+standort-hannover-Detail) · wissen (+wissen-artikel) · karriere ·
kontakt · glaeubiger-portal · impressum · datenschutz · schlichtung ·
barrierefreiheit · styleguide (intern, noindex)

## Vor Go-Live (Pflicht)
1. **⟨ Platzhalter ⟩ ersetzen**: Adressen, Telefonnummern, E-Mails,
   Registernummern, USt-IdNr, Versicherer, Kammern (impressum/datenschutz/
   schlichtung) — anwaltliche Endprüfung durch die Kanzlei.
2. **Fonts self-hosten**: woff2 nach `assets/fonts/`, Blöcke in
   `assets/fonts.css` aktivieren (aktuell System-Fallbacks; kein CDN).
3. **Kontaktformular anbinden**: `assets/site.js` zeigt derzeit nur das
   Erfolgspanel — Versand-Endpunkt (rate-limitiert, Double-Opt-in) ergänzen.
4. Karten-Koordinaten je Standort in `site.js` setzen (Klick-zum-Laden bleibt).
5. Gläubiger-Portal-CTA auf die Plattform-URL zeigen lassen (`/portal/`).

DSGVO-Eigenschaften: keine Cookies/kein Tracking → kein Banner nötig;
Karten laden erst auf Klick; externe Links mit noopener.
