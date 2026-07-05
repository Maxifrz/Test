/* ECKERT Website — schlankes Vanilla-JS (ersetzt die Prototyp-Runtime).
   Funktionen: Kontaktformular-Validierung + Erfolgsmeldung, Karte-auf-Klick
   (Datenschutz), rel-Härtung externer Links. Keine Cookies, kein Tracking. */
"use strict";
(function () {
  // ---------- Kontaktformular ----------
  var form = document.querySelector("form");
  var nameEl = document.getElementById("ek-name");
  if (form && nameEl) {
    var emailEl = document.getElementById("ek-email");
    var anliegenEl = document.getElementById("ek-anliegen");
    var consentEl = form.querySelector('input[type="checkbox"]');

    function setErr(el, msg) {
      if (!el) return !msg;
      el.setAttribute("aria-invalid", msg ? "true" : "false");
      el.style.borderColor = msg ? "var(--bad)" : "";
      var id = el.id + "-err";
      var node = document.getElementById(id);
      if (msg) {
        if (!node) {
          node = document.createElement("p");
          node.id = id;
          node.setAttribute("role", "alert");
          node.style.cssText = "margin:6px 0 0;font-size:12.5px;color:var(--bad);";
          el.insertAdjacentElement("afterend", node);
        }
        node.textContent = msg;
      } else if (node) {
        node.remove();
      }
      return !msg;
    }

    form.setAttribute("novalidate", "true");
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var ok = true;
      ok = setErr(nameEl, nameEl.value.trim() ? "" : "Bitte geben Sie Ihren Namen an.") && ok;
      ok = setErr(emailEl, /.+@.+\..+/.test(emailEl && emailEl.value || "") ? "" : "Bitte geben Sie eine gültige E-Mail-Adresse an.") && ok;
      ok = setErr(anliegenEl, anliegenEl && anliegenEl.value.trim().length >= 10 ? "" : "Bitte beschreiben Sie Ihr Anliegen (mind. 10 Zeichen).") && ok;
      if (consentEl && !consentEl.checked) {
        consentEl.style.outline = "2px solid var(--bad)";
        consentEl.style.outlineOffset = "2px";
        ok = false;
      } else if (consentEl) {
        consentEl.style.outline = "";
      }
      if (!ok) return;

      // Kein Backend konfiguriert → Erfolgspanel (Versand-Endpunkt vor Go-Live anbinden)
      var card = document.createElement("div");
      card.setAttribute("role", "status");
      card.style.cssText = "background:var(--card);border:1px solid var(--line);border-left:3px solid var(--ok);border-radius:2px;padding:28px 30px;";
      card.innerHTML =
        '<p style="font-family:var(--font-serif);font-size:22px;margin:0 0 8px;">Vielen Dank — Ihre Nachricht ist angekommen.</p>' +
        '<p style="font-size:14.5px;line-height:1.6;color:var(--ink-2);margin:0;">Wir melden uns innerhalb von 24 Stunden. In dringenden Fällen erreichen Sie die Krisen-Hotline unter ⟨ Telefonnummer ⟩.</p>';
      form.replaceWith(card);
      card.scrollIntoView({ block: "center", behavior: "smooth" });
    });
  }

  // ---------- Karte erst auf Klick laden (Datenschutz) ----------
  document.querySelectorAll("button, a").forEach(function (el) {
    if (el.textContent.trim().indexOf("Karte laden") !== 0) return;
    el.addEventListener("click", function (e) {
      e.preventDefault();
      var box = el.closest("div");
      var iframe = document.createElement("iframe");
      iframe.title = "Karte";
      iframe.loading = "lazy";
      iframe.style.cssText = "width:100%;height:100%;min-height:320px;border:0;display:block;";
      // Platzhalter-Koordinaten (Hannover Mitte) — vor Go-Live je Standort setzen
      iframe.src = "https://www.openstreetmap.org/export/embed.html?bbox=9.70%2C52.35%2C9.78%2C52.40&layer=mapnik";
      if (box) { box.innerHTML = ""; box.appendChild(iframe); }
    });
  });

  // ---------- Mobile Navigation (Burger, ≤920px via responsive.css) ----------
  var header = document.querySelector("header");
  var nav = header && header.querySelector('nav[aria-label="Hauptnavigation"]');
  if (header && nav) {
    var bar = nav.parentElement;
    var burger = document.createElement("button");
    burger.className = "ek-burger";
    burger.setAttribute("aria-label", "Menü öffnen");
    burger.setAttribute("aria-expanded", "false");
    burger.innerHTML =
      '<svg width="20" height="20" viewBox="0 0 20 20" stroke="currentColor" stroke-width="1.6" fill="none">' +
      '<path d="M3 5h14M3 10h14M3 15h14"/></svg>';

    var menu = document.createElement("nav");
    menu.className = "ek-mobilemenu";
    menu.setAttribute("aria-label", "Mobile Navigation");
    nav.querySelectorAll("a").forEach(function (a) {
      menu.appendChild(a.cloneNode(true));
    });
    // Krisen-Hotline-CTA (falls vorhanden) ans Menüende
    var cta = bar.querySelector('a[href*="kontakt"]');
    if (cta && /Hotline/i.test(cta.textContent)) {
      var m = cta.cloneNode(true);
      m.removeAttribute("style");
      m.style.cssText = "display:block;padding:13px 4px;font-weight:600;color:var(--accent);text-decoration:none;border-bottom:none;";
      menu.appendChild(m);
    }

    bar.appendChild(burger);
    header.style.position = "sticky";
    header.appendChild(menu);

    burger.addEventListener("click", function () {
      var open = menu.classList.toggle("open");
      burger.setAttribute("aria-expanded", open ? "true" : "false");
      burger.setAttribute("aria-label", open ? "Menü schließen" : "Menü öffnen");
    });
    menu.addEventListener("click", function (e) {
      if (e.target.closest("a")) menu.classList.remove("open");
    });
  }

  // ---------- Externe Links härten ----------
  document.querySelectorAll('a[href^="http"]').forEach(function (a) {
    if (a.hostname !== location.hostname) a.rel = "noopener noreferrer";
  });
})();
